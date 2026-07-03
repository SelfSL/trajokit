"""Verifier seam: reward = pluggable scoring of a finished episode.

- TestCmdVerifier: run the env's own test command (SWE-bench style, verifiable)
- JudgeVerifier: RLAIF — an LLM judge scores the transcript (customer-service style)

Parse/infra failures RAISE (=> orchestrator drops the rollout) instead of returning 0,
consistent with env-failure != task-failure.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from .judges import JudgeClient
from .sandbox import Sandbox
from .types import Task

DEFAULT_RUBRIC = (
    "You are grading one episode of an autonomous agent solving a task.\n"
    "Judge ONLY whether the task was actually accomplished, based on the transcript.\n"
    'Respond with ONLY this JSON: {"score": <float between 0.0 and 1.0>}'
)

_JSON_RE = re.compile(r"\{[^{}]*\}")


class Verifier(Protocol):
    async def score(self, task: Task, sandbox: Sandbox, transcript: str) -> float: ...


class TestCmdVerifier:
    async def score(self, task: Task, sandbox: Sandbox, transcript: str) -> float:
        test_cmd = task.env_spec.get("test_cmd")
        if not test_cmd:
            return 0.0
        res = await sandbox.exec(test_cmd, timeout=float(task.env_spec.get("test_timeout", 600)))
        return 1.0 if res.returncode == 0 else 0.0


class JudgeVerifier:
    def __init__(self, client: JudgeClient, rubric: str = DEFAULT_RUBRIC,
                 max_transcript_chars: int = 60_000):
        self.client = client
        self.rubric = rubric
        self.max_chars = max_transcript_chars

    async def score(self, task: Task, sandbox: Sandbox, transcript: str) -> float:
        user = f"TASK:\n{task.prompt}\n\nEPISODE TRANSCRIPT:\n{transcript[-self.max_chars:]}"
        text = await self.client.complete(self.rubric, user)
        m = _JSON_RE.search(text)
        if not m:
            raise ValueError(f"judge returned unparseable output: {text[:200]!r}")
        score = float(json.loads(m.group(0))["score"])
        return min(1.0, max(0.0, score))
