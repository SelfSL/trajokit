"""cline-bench -> trajokit Tasks.

Maps a local clone of github.com/cline/cline-bench onto the standard Task shape:
  prompt    = instruction.md, verbatim
  image     = built locally from each task's environment/Dockerfile
  test_cmd  = self-contained verifier: injects the task's tests/ into the container
              (base64 tarball embedded in the command) and runs them. Tests are
              therefore NEVER present during the episode, only at scoring time.
  oracle_cmd (extra) = injects solution/ and runs solve.sh, enabling a positive
              control (oracle must score 1.0) without any harness code.

Resource/timeout hints come from task.toml ([verifier].timeout_sec etc.).
Harbor/Daytona/Cline-CLI from the upstream harness are bypassed entirely.

CLI:
    python -m trajokit.datasets.clinebench --clone-dir /path/to/cline-bench \
        --out clinebench.jsonl [--limit N] [--skip-build]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import tarfile
import tomllib
from pathlib import Path

from ..types import Task

INJECT_DIR_TESTS = "/tests"  # hardcoded by cline-bench test.sh convention
INJECT_DIR_SOLUTION = "/trajokit_solution"


def _last_workdir(dockerfile_text: str, default: str = "/app") -> str:
    wd = default
    for line in dockerfile_text.splitlines():
        s = line.strip()
        if s.upper().startswith("WORKDIR "):
            wd = s.split(None, 1)[1].strip()
    return wd


def _dir_b64(d: Path) -> str:
    """Deterministic gzipped tarball of a directory, base64-encoded."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(d)))
    return base64.b64encode(buf.getvalue()).decode()


def _inject_cmd(b64: str, dest: str) -> str:
    return (
        f"rm -rf {dest} && mkdir -p {dest} && "
        f"printf '%s' {b64} | base64 -d | tar -xzf - -C {dest}"
    )


def _test_cmd(tests_dir: Path, workdir: str) -> str:
    b64 = _dir_b64(tests_dir)
    if (tests_dir / "test.sh").exists():
        # Canonical cline-bench runner: always exits 0 and writes the verdict to
        # /logs/verifier/reward.txt, so gate the exit code on that file's content.
        run = (
            "mkdir -p /logs/verifier && bash /tests/test.sh; "
            "[ \"$(cat /logs/verifier/reward.txt 2>/dev/null)\" = \"1\" ]"
        )
    elif (tests_dir / "run-tests.sh").exists():
        run = f"cd {workdir} && bash {INJECT_DIR_TESTS}/run-tests.sh"
    else:
        run = (
            f"cd {workdir} && "
            f"(uv run pytest -q {INJECT_DIR_TESTS} 2>/dev/null || pytest -q {INJECT_DIR_TESTS})"
        )
    return f"{_inject_cmd(b64, INJECT_DIR_TESTS)} && {run}"


def _oracle_cmd(solution_dir: Path, workdir: str) -> str:
    b64 = _dir_b64(solution_dir)
    return (
        f"{_inject_cmd(b64, INJECT_DIR_SOLUTION)} && "
        f"cd {workdir} && bash {INJECT_DIR_SOLUTION}/solve.sh"
    )


def _image_exists(tag: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", tag], capture_output=True
    ).returncode == 0


def _build_image(task_dir: Path, tag: str, timeout: float, retries: int = 1) -> None:
    """Build with one retry (Dockerfiles fetch from the network; flakes happen).

    Already-built tags are reused, so reruns after a partial failure are cheap.
    """
    if _image_exists(tag):
        return
    last = None
    for attempt in range(retries + 1):
        r = subprocess.run(
            ["docker", "build", "-t", tag, str(task_dir / "environment")],
            capture_output=True, timeout=timeout, text=True,
        )
        if r.returncode == 0:
            return
        last = r
    raise RuntimeError(
        f"docker build failed for {task_dir.name} after {retries + 1} attempts; "
        f"stderr tail:\n{(last.stderr or last.stdout)[-1500:]}"
    )


def load_clinebench(clone_dir: str | Path, limit: int | None = None,
                    build: bool = True, skip_failed: bool = False) -> list[Task]:
    root = Path(clone_dir) / "tasks"
    if not root.is_dir():
        raise FileNotFoundError(f"no tasks/ under {clone_dir}; clone cline-bench first")

    tasks: list[Task] = []
    failed: list[str] = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg = tomllib.loads((task_dir / "task.toml").read_text())
        dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
        workdir = _last_workdir(dockerfile)
        tag = f"trajokit-clinebench:{task_dir.name.split('-')[0]}"

        if build:
            try:
                _build_image(task_dir, tag,
                             float(cfg.get("environment", {}).get("build_timeout_sec", 900)))
            except Exception as e:
                if skip_failed:
                    print(f"SKIP {task_dir.name}: {e}")
                    failed.append(task_dir.name)
                    continue
                raise

        env_spec = {
            "image": tag,
            "workdir": workdir,
            "network": "bridge",  # verifier (uv pip) and oracle (git/bun) need egress
            "test_cmd": _test_cmd(task_dir / "tests", workdir),
            "test_timeout": float(cfg.get("verifier", {}).get("timeout_sec", 1800)),
            "agent_timeout": float(cfg.get("agent", {}).get("timeout_sec", 1800)),
        }
        sol = task_dir / "solution"
        if (sol / "solve.sh").exists():
            env_spec["oracle_cmd"] = _oracle_cmd(sol, workdir)

        tasks.append(
            Task(
                task_id=task_dir.name,
                prompt=(task_dir / "instruction.md").read_text(),
                env_spec=env_spec,
            )
        )
        if limit and len(tasks) >= limit:
            break
    if failed:
        print(f"{len(failed)} task(s) skipped due to build failures: {failed}")
    return tasks


def write_jsonl(tasks: list[Task], path: str | Path) -> None:
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps({"task_id": t.task_id, "prompt": t.prompt,
                                "env_spec": t.env_spec}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone-dir", required=True, help="path to a cline-bench checkout")
    ap.add_argument("--out", default="clinebench.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-build", action="store_true",
                    help="assume images already built (reuse tags)")
    ap.add_argument("--skip-failed", action="store_true",
                    help="skip tasks whose image build fails instead of aborting")
    args = ap.parse_args()
    ts = load_clinebench(args.clone_dir, limit=args.limit, build=not args.skip_build,
                         skip_failed=args.skip_failed)
    write_jsonl(ts, args.out)
    print(f"wrote {len(ts)} tasks -> {args.out}")
