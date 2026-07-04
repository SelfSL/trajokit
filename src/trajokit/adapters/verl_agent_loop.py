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
        return {
            "text": self.tok.decode(out.token_ids),
            "token_ids": list(out.token_ids),
            "finish_reason": out.stop_reason,
        }


def _make_agent_loop_cls():
    """Build the verl subclass lazily (verl import only when actually used)."""
    from verl.experimental.agent_loop.agent_loop import (  # noqa: PLC0415
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )

    @register("trajokit")
    class TrajokitAgentLoop(AgentLoopBase):
        async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
            t0 = time.time()
            env_spec = kwargs["env_spec"]
            task = Task(
                task_id=str(kwargs.get("task_id", kwargs.get("index", uuid.uuid4().hex))),
                prompt=kwargs["prompt"] if isinstance(kwargs["prompt"], str) else kwargs["prompt"][-1]["content"],
                env_spec=json.loads(env_spec) if isinstance(env_spec, str) else env_spec,
            )
            tk = self.rollout_config.get("trajokit", {}) or {}
            loop = AgentLoop(
                tokenizer=self.tokenizer,
                max_turns=int(tk.get("max_turns", 30)),
                max_context=int(tk.get("max_context", self.rollout_config.get("max_model_len", 32768))),
                max_obs_chars=int(tk.get("max_obs_chars", 8000)),
                gen_kwargs={"temperature": sampling_params.get("temperature", 1.0)},
                chat_template_kwargs=self.apply_chat_template_kwargs,
            )
            policy = VerlPolicyShim(self.server_manager, self.tokenizer,
                                    request_id=uuid.uuid4().hex,
                                    base_sampling_params=sampling_params)
            async with _SANDBOX_SEM:
                traj = await loop.run(task, policy, LocalDockerSandbox())

            prompt_ids, response_ids, response_mask = split_prompt_response(traj)
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                reward_score=traj.reward,
                num_turns=traj.info["turns"],
                metrics=AgentLoopMetrics(generate_sequences=time.time() - t0),
                extra_fields={"task_id": traj.task_id, "patch": traj.info.get("patch", ""),
                              "submitted": traj.info.get("submitted", False)},
            )

    return TrajokitAgentLoop


try:  # register on import when verl is present; harmless no-op otherwise
    TrajokitAgentLoop = _make_agent_loop_cls()
except ImportError:  # pragma: no cover
    TrajokitAgentLoop = None
