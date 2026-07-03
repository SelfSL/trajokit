"""Orchestrator: run GRPO groups with bounded concurrency.

Key policy: ENV FAILURE != TASK FAILURE. A crashed container or sandbox error drops
the rollout (returns None) instead of scoring 0 — scoring infra flakiness poisons
the reward signal silently.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from .loop import AgentLoop
from .policy import PolicyClient
from .sandbox import Sandbox
from .types import Task, Trajectory

log = logging.getLogger("trajokit")


class Orchestrator:
    def __init__(
        self,
        loop: AgentLoop,
        policy: PolicyClient,
        sandbox_factory: Callable[[], Sandbox],
        max_concurrency: int = 64,
    ):
        self.loop = loop
        self.policy = policy
        self.sandbox_factory = sandbox_factory
        self.sem = asyncio.Semaphore(max_concurrency)

    async def _one(self, task: Task) -> Trajectory | None:
        async with self.sem:
            try:
                return await self.loop.run(task, self.policy, self.sandbox_factory())
            except Exception as e:  # noqa: BLE001 — env failure => drop, never reward 0
                log.warning("dropping rollout for %s: %r", task.task_id, e)
                return None

    async def run_group(self, task: Task, k: int) -> list[Trajectory]:
        """K rollouts of one task (a GRPO/GSPO group)."""
        results = await asyncio.gather(*[self._one(task) for _ in range(k)])
        return [t for t in results if t is not None]

    async def run_batch(self, tasks: list[Task], k: int) -> dict[str, list[Trajectory]]:
        groups = await asyncio.gather(*[self.run_group(t, k) for t in tasks])
        return {t.task_id: g for t, g in zip(tasks, groups)}
