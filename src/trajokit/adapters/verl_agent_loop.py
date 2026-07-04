"""Run trajokit episodes inside upstream verl's agent-loop rollout.

Wiring:
  verl AgentLoopWorker -> TrajokitAgentLoop.run(...)
      -> trajokit AgentLoop (owns tokenization/masks/sandbox/reward)
      -> policy = VerlPolicyShim over verl's LLMServerClient.generate (token-in/token-out)
      -> Trajectory -> AgentLoopOutput (prompt_ids/response_ids/response_mask/reward_score)

verl imports are lazy so trajokit stays importable (and testable) without verl.

Enable in verl via an agent-loop config yaml:
    - name: trajokit
      _target_: trajokit.adapters.verl_agent_loop.TrajokitAgentLoop
and a dataset with non-tensor fields: task_id, prompt, env_spec (JSON string),
agent_name="trajokit".
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

from ..cache import cache_from_env
from ..loop import AgentLoop
from ..sandbox import LocalDockerSandbox
from ..types import Task
from .verl import split_prompt_response

_SANDBOX_SEM = asyncio.Semaphore(int(os.environ.get("TRAJOKIT_MAX_SANDBOXES", "64")))


class VerlPolicyShim:
    """trajokit policy interface backed by verl's LLMServerClient (token-in/token-out).

    Using token ids end-to-end preserves trajokit's no-retokenization guarantee.
    """

    def __init__(self, server_manager: Any, tokenizer: Any, request_id: str,
                 base_sampling_params: dict[str, Any] | None = None):
        self.server_manager = server_manager
        self.tok = tokenizer
        self.request_id = request_id  # sticky session -> same server across turns (prefix cache)
        self.base = dict(base_sampling_params or {})
        self.min_global_steps: int | None = None  # weight-version range across turns
        self.max_global_steps: int | None = None

    async def complete(self, prompt_token_ids: list[int], max_tokens: int = 2048,
                       temperature: float | None = None, stop: list[str] | None = None) -> dict:
        sp = dict(self.base)
        sp["max_tokens"] = max_tokens
        if temperature is not None:
            sp["temperature"] = temperature
        if stop:
            sp["stop"] = stop
        out = await self.server_manager.generate(
            request_id=self.request_id, prompt_ids=prompt_token_ids, sampling_params=sp
        )
        ef = getattr(out, "extra_fields", None) or {}
        mn, mx = ef.get("min_global_steps"), ef.get("max_global_steps")
        if mn is not None:
            self.min_global_steps = mn if self.min_global_steps is None else min(self.min_global_steps, mn)
        if mx is not None:
            self.max_global_steps = mx if self.max_global_steps is None else max(self.max_global_steps, mx)
        return {
            "text": self.tok.decode(out.token_ids),
            "token_ids": list(out.token_ids),
            "logprobs": list(out.log_probs) if getattr(out, "log_probs", None) else None,
            "finish_reason": out.stop_reason,
        }


try:  # verl is optional: VerlPolicyShim stays importable/testable without it
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )

    _HAS_VERL = True
except ImportError:  # pragma: no cover
    _HAS_VERL = False


if _HAS_VERL:

    @register("trajokit")
    class TrajokitAgentLoop(AgentLoopBase):
        # NOTE: must be a top-level class — hydra instantiates by qualname, and
        # classes defined inside functions get '<locals>' qualnames it cannot import.

        async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
            t0 = time.time()
            env_spec = kwargs["env_spec"]
            task = Task(
                task_id=str(kwargs.get("task_id", kwargs.get("index", uuid.uuid4().hex))),
                prompt=kwargs["prompt"] if isinstance(kwargs["prompt"], str) else kwargs["prompt"][-1]["content"],
                env_spec=json.loads(env_spec) if isinstance(env_spec, str) else env_spec,
            )
            loop = AgentLoop(
                tokenizer=self.tokenizer,
                max_turns=int(os.environ.get("TRAJOKIT_MAX_TURNS", "30")),
                max_context=int(os.environ.get("TRAJOKIT_MAX_CONTEXT", "32768")),
                max_obs_chars=int(os.environ.get("TRAJOKIT_MAX_OBS_CHARS", "8000")),
                gen_kwargs={"temperature": sampling_params.get("temperature", 1.0)},
                chat_template_kwargs=self.apply_chat_template_kwargs,
            )
            policy = VerlPolicyShim(self.server_manager, self.tokenizer,
                                    request_id=uuid.uuid4().hex,
                                    base_sampling_params=sampling_params)
            cache = cache_from_env()
            traj = cache.load_one(task.task_id) if cache else None
            if traj is None:
                async with _SANDBOX_SEM:
                    traj = await loop.run(task, policy, LocalDockerSandbox())
                if cache:
                    cache.save(traj)

            prompt_ids, response_ids, response_mask = split_prompt_response(traj)
            response_logprobs = traj.logprobs[len(prompt_ids):] if traj.logprobs else None
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                reward_score=traj.reward,
                num_turns=traj.info["turns"],
                metrics=AgentLoopMetrics(generate_sequences=time.time() - t0),
                extra_fields={"task_id": traj.task_id, "patch": traj.info.get("patch", ""),
                              "submitted": traj.info.get("submitted", False),
                              # weight-version tags verl's staleness metrics expect;
                              # -1 = unknown (e.g. cache replay)
                              "min_global_steps": -1 if policy.min_global_steps is None else policy.min_global_steps,
                              "max_global_steps": -1 if policy.max_global_steps is None else policy.max_global_steps},
            )
