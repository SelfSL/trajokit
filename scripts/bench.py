import argparse
import asyncio
import time

from transformers import AutoTokenizer
from trajokit import AgentLoop, Orchestrator, PolicyClient, LocalDockerSandbox, load_tasks


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                    help="served model name")
    ap.add_argument("--tokenizer", default="/mnt/raid5/models/Qwen3-Coder-30B-A3B-Instruct",
                    help="tokenizer path (defaults to --model if empty)")
    ap.add_argument("--tasks", default="tasks.jsonl")
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--k", type=int, default=4, help="rollouts per task (GRPO group size)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--max-context", type=int, default=32768)
    return ap.parse_args()


async def main(args):
    tasks = load_tasks(args.tasks)[: args.n_tasks]
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    loop = AgentLoop(tokenizer=tok, max_turns=args.max_turns, max_context=args.max_context)
    policy = PolicyClient("http://localhost:8000", model=args.model)
    orch = Orchestrator(loop, policy, sandbox_factory=LocalDockerSandbox,
                        max_concurrency=args.concurrency)

    t0 = time.time()
    groups = await orch.run_batch(tasks, k=args.k)
    wall = time.time() - t0

    total = sum(len(g) for g in groups.values())
    for tid, g in groups.items():
        print(f"{tid}: rewards={[t.reward for t in g]} turns={[t.info['turns'] for t in g]}")
    print(f"\n{total}/{args.n_tasks * args.k} rollouts survived | wall={wall/60:.1f} min "
          f"| {wall/max(total,1):.0f}s/rollout effective")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
