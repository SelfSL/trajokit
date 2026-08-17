"""cline-bench controls: verify the verifier before trusting any agent number.

  --mode neg : untouched container, run test_cmd            -> expect exit != 0
  --mode pos : run oracle_cmd (solution/solve.sh), then test_cmd -> expect exit == 0

Usage:
    uv run python scripts/clinebench_control.py --tasks clinebench.jsonl --task-idx 0 --mode neg
    uv run python scripts/clinebench_control.py --tasks clinebench.jsonl --task-idx 0 --mode pos
"""
import argparse
import asyncio

from trajokit import LocalDockerSandbox, load_tasks


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="clinebench.jsonl")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--mode", choices=["neg", "pos"], required=True)
    return ap.parse_args()


async def main(args):
    task = load_tasks(args.tasks)[args.task_idx]
    sb = LocalDockerSandbox()
    await sb.start(task.env_spec)
    try:
        if args.mode == "pos":
            oracle = task.env_spec.get("oracle_cmd")
            if not oracle:
                print("no oracle_cmd for this task; cannot run positive control")
                return
            r = await sb.exec(oracle, timeout=task.env_spec.get("test_timeout", 1800))
            print(f"oracle exit: {r.returncode}")
            if r.returncode != 0:
                print("ORACLE FAILED. stderr tail:\n", r.stderr[-1500:])
                return
        res = await sb.exec(task.env_spec["test_cmd"],
                            timeout=task.env_spec.get("test_timeout", 1800))
        expect = "0 (pass)" if args.mode == "pos" else "!= 0 (fail)"
        print(f"[{task.task_id}] mode={args.mode} test exit: {res.returncode} (expected {expect})")
        print("STDOUT tail:\n", res.stdout[-1500:])
        print("STDERR tail:\n", res.stderr[-800:])
    finally:
        await sb.stop()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
