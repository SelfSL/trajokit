"""SWE-bench Verified -> trajokit Tasks.

Uses official prebuilt x86_64 images from Docker Hub (`swebench/` namespace) and the
`swebench` package's per-repo test specs.

POC reward simplification: apply the gold test_patch, then run only FAIL_TO_PASS
tests; exit code 0 => reward 1.0. (No PASS_TO_PASS regression check, no log parsing.)
Official harness evaluation stays the source of truth for reported numbers.

CLI:
    python -m trajokit.datasets.swebench --out tasks.jsonl --limit 10
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path

from ..types import Task

PROMPT_TMPL = (
    "Repository is checked out at /testbed with all dependencies installed.\n"
    "Fix the issue described below by editing source files (do NOT modify tests).\n\n"
    "<issue>\n{problem}\n</issue>"
)


def _image(instance_id: str) -> str:
    # official naming escapes '__' as '_1776_'
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


def _test_cmd(row: dict) -> str:
    """apply gold test_patch, then run FAIL_TO_PASS with the repo's own test command."""
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

    spec = MAP_REPO_VERSION_TO_SPECS[row["repo"]][row["version"]]
    f2p = json.loads(row["FAIL_TO_PASS"]) if isinstance(row["FAIL_TO_PASS"], str) else row["FAIL_TO_PASS"]
    tests = " ".join(shlex.quote(t) for t in f2p)
    # heredoc terminator must be alone on its line; chain via set -e, not '&&'
    return (
        "set -e\n"
        "cd /testbed\n"
        "git apply -v - <<'TRAJOKIT_EOF'\n"
        f"{row['test_patch']}\n"
        "TRAJOKIT_EOF\n"
        f"{spec['test_cmd']} {tests}\n"
    )


def load_swebench_verified(limit: int | None = None, split: str = "test") -> list[Task]:
    from datasets import load_dataset  # optional dep: trajokit[swebench]

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split=split)
    tasks = []
    for row in ds:
        tasks.append(
            Task(
                task_id=row["instance_id"],
                prompt=PROMPT_TMPL.format(problem=row["problem_statement"]),
                env_spec={
                    "image": _image(row["instance_id"]),
                    "workdir": "/testbed",
                    "test_cmd": _test_cmd(row),
                    "test_timeout": 900,
                },
            )
        )
        if limit and len(tasks) >= limit:
            break
    return tasks


def write_jsonl(tasks: list[Task], path: str | Path) -> None:
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps({"task_id": t.task_id, "prompt": t.prompt, "env_spec": t.env_spec}) + "\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="swebench_verified.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tasks = load_swebench_verified(limit=args.limit)
    write_jsonl(tasks, args.out)
    print(f"wrote {len(tasks)} tasks -> {args.out}")
