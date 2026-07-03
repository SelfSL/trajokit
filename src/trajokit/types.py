"""Core data types. Everything is data; only the loop is code."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Task:
    """One RL task instance. Benchmark-agnostic: a new benchmark is a new loader, not new code."""
    task_id: str
    prompt: str                      # fully rendered instruction shown to the agent
    env_spec: dict[str, Any]         # e.g. {"image": ..., "test_cmd": ..., "workdir": ...}


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


@dataclass
class Trajectory:
    """What trainers consume. loss_mask: 1 = assistant-generated token, 0 = prompt/tool tokens."""
    task_id: str
    input_ids: list[int]
    loss_mask: list[int]
    reward: float
    info: dict[str, Any] = field(default_factory=dict)
    turn_spans: list[tuple[int, int]] = field(default_factory=list)  # [start, end) per assistant turn

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.loss_mask):
            raise ValueError(
                f"input_ids ({len(self.input_ids)}) and loss_mask ({len(self.loss_mask)}) length mismatch"
            )


def load_tasks(jsonl_path: str | Path) -> list[Task]:
    tasks = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                tasks.append(Task(task_id=d["task_id"], prompt=d["prompt"], env_spec=d["env_spec"]))
    return tasks
