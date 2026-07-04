"""AgentLoop: the package's heart.

Design invariants:
1. INCREMENTAL TOKENIZATION — the running token buffer is append-only. We never
   re-render or re-tokenize the full conversation (retokenization drift is the #1
   silent mask bug in agentic RL).
2. Assistant tokens come back from the server as token ids (mask=1). Env output is
   templated as a user turn and tokenized once (mask=0).
3. Bash-agent format: the model emits ```bash ...``` blocks; `submit` ends the episode.
"""
from __future__ import annotations

import re
import time
import shlex
from typing import Any

from .policy import PolicyClient
from .sandbox import Sandbox
from .types import Task, Trajectory
from .verifiers import TestCmdVerifier, Verifier

FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
BASH_RE = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)
SUBMIT_RE = re.compile(r"^\s*submit\s*$", re.MULTILINE | re.IGNORECASE)

SYSTEM = (
    "You are an autonomous software engineer working in a sandboxed repo.\n"
    "Each turn: think briefly, then EITHER emit exactly one ```bash ...``` block "
    "to run a command, OR write `submit` on its own line when the task is complete."
    "Commands must be non-interactive (no nano/vim); edit files via sed or cat/python heredocs.\n"
)


class AgentLoop:
    def __init__(
        self,
        tokenizer: Any,                      # HF tokenizer (must match the served model!)
        max_turns: int = 30,
        max_context: int = 32768,
        max_obs_chars: int = 8000,
        gen_kwargs: dict | None = None,
        verifier: Verifier | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        self.ct_kwargs = chat_template_kwargs or {}
        self.verifier = verifier or TestCmdVerifier()
        self.tok = tokenizer
        self.max_turns = max_turns
        self.max_context = max_context
        self.max_obs_chars = max_obs_chars
        self.gen_kwargs = gen_kwargs or {}

    # ---- template fragments (tokenized once, appended; never re-rendered) ----

    def _prefix_ids(self, task: Task) -> list[int]:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task.prompt}]
        out = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        ids = out if isinstance(out, list) else out["input_ids"]  # transformers 5.x: BatchEncoding
        if ids and isinstance(ids[0], list):  # possible batch dim
            ids = ids[0]
        return list(ids)

    def _obs_ids(self, obs: str, drop_assistant_close: bool = False) -> list[int]:
        """Tokenize env output as a template delta — never re-render the full conversation.

        drop_assistant_close: True when the model already generated its own
        end-of-turn token, so we must not add a duplicate close.
        """
        # render 3 snapshots of a dummy conversation with the tokenizer's own template:
        # base = "...<|im_start|>assistant\n"                      (before generation)
        base = self.tok.apply_chat_template(
            [{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False
        )
        # mid  = base + "<|im_end|>\n"                             (assistant turn closed)
        mid = self.tok.apply_chat_template(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": ""}], tokenize=False
        )
        # full = mid + "<|im_start|>user\n{obs}<|im_end|>\n<|im_start|>assistant\n"
        full = self.tok.apply_chat_template(
            [{"role": "user", "content": "x"},
             {"role": "assistant", "content": ""},
             {"role": "user", "content": obs}],
            add_generation_prompt=True, tokenize=False,
        )
        # what we need to append after the model's turn:
        delta = full[len(base):]
        if drop_assistant_close:
            close = mid[len(base):]            # the "<|im_end|>\n" fragment
            if close and delta.startswith(close):
                delta = delta[len(close):]     # model already emitted it — skip duplicate
        return self.tok(delta, add_special_tokens=False)["input_ids"]

    # ---- main loop ----

    async def run(self, task: Task, policy: PolicyClient, sandbox: Sandbox) -> Trajectory:
        t0 = time.time()
        ids: list[int] = list(self._prefix_ids(task))
        mask: list[int] = [0] * len(ids)
        spans: list[tuple[int, int]] = []
        transcript: list[str] = [f"[task]\n{task.prompt}"]
        turns, truncated, submitted = 0, False, False

        await sandbox.start(task.env_spec)
        try:
            while turns < self.max_turns:
                budget = self.max_context - len(ids) - 64
                if budget <= 256:
                    truncated = True
                    break

                out = await policy.complete(
                    prompt_token_ids=ids,
                    max_tokens=min(2048, budget),
                    **self.gen_kwargs,
                )
                gen_ids = out["token_ids"]
                if gen_ids is None:  # server didn't return ids; tokenize text (drift risk, warn)
                    gen_ids = self.tok(out["text"], add_special_tokens=False)["input_ids"]
                spans.append((len(ids), len(ids) + len(gen_ids)))
                ids += gen_ids
                mask += [1] * len(gen_ids)
                turns += 1

                text = out["text"]
                transcript.append(f"[agent]\n{text}")
                if SUBMIT_RE.search(FENCE_RE.sub("", text)):
                    submitted = True
                    break
                m = BASH_RE.search(text)
                if m:
                    res = await sandbox.exec(m.group(1).strip())
                    obs = (res.stdout + res.stderr)[-self.max_obs_chars:]
                    obs = f"(exit={res.returncode})\n{obs}" if not res.timed_out else "(TIMEOUT)"
                else:
                    obs = "No ```bash``` block found. Emit one command, or `submit`."

                transcript.append(f"[env]\n{obs}")
                # did the model end its turn with its own eos (<|im_end|>)?
                ended_eos = bool(gen_ids) and gen_ids[-1] == getattr(self.tok, "eos_token_id", None)
                # if so, the obs delta must not re-add the close token
                obs_ids = self._obs_ids(obs, drop_assistant_close=ended_eos)
                ids += obs_ids
                mask += [0] * len(obs_ids)

            patch = ""
            workdir = task.env_spec.get("workdir")
            if workdir:  # capture model patch BEFORE verifier mutates the repo
                res = await sandbox.exec(f"git -C {shlex.quote(workdir)} diff", timeout=60)
                if res.returncode == 0:
                    patch = res.stdout
            reward = await self.verifier.score(task, sandbox, "\n\n".join(transcript))
        finally:
            await sandbox.stop()

        return Trajectory(
            task_id=task.task_id,
            input_ids=ids,
            loss_mask=mask,
            reward=reward,
            info={
                "turns": turns,
                "truncated": truncated,
                "submitted": submitted,
                "wall_s": round(time.time() - t0, 1),
                "patch": patch,
            },
            turn_spans=spans,
        )
