"""Disk cache for rollout trajectories: generate once, replay instantly.

Enable via env:
    TRAJOKIT_ROLLOUT_CACHE=/path/to/cache      # turn caching on
    TRAJOKIT_CACHE_RUN=<any-id>                # claim namespace (new id => files reusable)

Each saved trajectory is one JSON file under <cache>/<task_id>/. A reader claims a
file by atomically creating "<file>.claim.<run_id>" (O_CREAT|O_EXCL), so concurrent
workers never hand out the same rollout twice within a run, and a fresh run_id makes
the whole cache replayable again.

CAUTION: cached rollouts are off-policy after any weight update. Use for debugging
the training path, seeding replay buffers, or SFT data - not fresh on-policy steps.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

from .types import Trajectory


class RolloutCache:
    def __init__(self, root: str | Path, run_id: str | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or os.environ.get("TRAJOKIT_CACHE_RUN", "default")

    def _dir(self, task_id: str) -> Path:
        return self.root / task_id.replace("/", "_")

    def save(self, traj: Trajectory) -> Path:
        d = self._dir(traj.task_id)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{uuid.uuid4().hex}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(traj)))
        tmp.replace(p)  # atomic: never a half-written cache file
        return p

    def load_one(self, task_id: str) -> Trajectory | None:
        """Claim and return one unclaimed cached trajectory for this task, else None."""
        d = self._dir(task_id)
        if not d.is_dir():
            return None
        for p in sorted(d.glob("*.json")):
            claim = p.with_name(p.name + f".claim.{self.run_id}")
            try:
                fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                continue  # already handed out in this run
            data = json.loads(p.read_text())
            data["turn_spans"] = [tuple(s) for s in data.get("turn_spans", [])]
            return Trajectory(**data)
        return None


def cache_from_env() -> RolloutCache | None:
    root = os.environ.get("TRAJOKIT_ROLLOUT_CACHE")
    return RolloutCache(root) if root else None

