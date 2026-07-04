import argparse
import asyncio
import json
import time

from transformers import AutoTokenizer
from trajokit import AgentLoop, Orchestrator, PolicyClient, LocalDockerSandbox, load_tasks


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--tokenizer", default="/mnt/raid5/models/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--tasks", default="tasks_100.jsonl")
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=50)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--out", default="predictions.jsonl")
    return ap.parse_args()


async def main(args):
    tasks = load_tasks(args.tasks)[: args.n_tasks]
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    loop = AgentLoop(tokenizer=tok, max_turns=args.max_turns, max_context=args.max_context,
                     gen_kwargs={"temperature": 0.0})  # greedy pass@1
    policy = PolicyClient("http://localhost:8000", model=args.model)
    orch = Orchestrator(loop, policy, sandbox_factory=LocalDockerSandbox,
                        max_concurrency=args.concurrency)

    t0, done = time.time(), 0

    async def run_one(task):
        nonlocal done
        group = await orch.run_group(task, k=1)
        done += 1
        patch_ok = bool(group and group[0].info.get("patch"))
        mark = "✅" if group and group[0].reward > 0 else "  "
        print(f"{mark} [{done}/{len(tasks)} {time.time()-t0:5.0f}s] {task.task_id} "
              f"turns={group[0].info['turns'] if group else '-'} "
              f"patch={'y' if patch_ok else 'EMPTY'}", flush=True)
        return task.task_id, group

    results = await asyncio.gather(*[run_one(t) for t in tasks])

    with open(args.out, "w") as f:
        for tid, g in results:
            patch = g[0].info.get("patch", "") if g else ""
            f.write(json.dumps({"instance_id": tid, "model_name_or_path": args.model,
                                "model_patch": patch}) + "\n")
    print(f"\nwrote {args.out}; grade with:\n"
          f"  uv run python -m swebench.harness.run_evaluation "
          f"--dataset_name princeton-nlp/SWE-bench_Verified "
          f"--predictions_path {args.out} --max_workers 8 --run_id trajokit-100")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
