import argparse
import asyncio
import time

from transformers import AutoTokenizer
from trajokit import AgentLoop, Orchestrator, PolicyClient, LocalDockerSandbox, load_tasks


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--tokenizer", default="/mnt/raid5/models/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--tasks", default="tasks.jsonl")
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=1.0)
    return ap.parse_args()


async def main(args):
    tasks = load_tasks(args.tasks)[: args.n_tasks]
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    loop = AgentLoop(tokenizer=tok, max_turns=args.max_turns, max_context=args.max_context,
                     gen_kwargs={"temperature": args.temperature})
    policy = PolicyClient("http://localhost:8000", model=args.model)
    orch = Orchestrator(loop, policy, sandbox_factory=LocalDockerSandbox,
                        max_concurrency=args.concurrency)

    t0 = time.time()

    async def run_one(task):
        group = await orch.run_group(task, args.k)
        rewards = [t.reward for t in group]
        turns = [t.info["turns"] for t in group]
        mark = "✅" if any(r > 0 for r in rewards) else "  "
        print(f"{mark} [{time.time()-t0:6.0f}s] {task.task_id}: rewards={rewards} turns={turns}",
              flush=True)
        return task.task_id, group

    results = await asyncio.gather(*[run_one(t) for t in tasks])

    wall = time.time() - t0
    all_trajs = [t for _, g in results for t in g]
    solved = sum(1 for _, g in results if any(t.reward > 0 for t in g))
    print(f"\n{len(all_trajs)}/{args.n_tasks * args.k} rollouts survived | "
          f"{solved}/{args.n_tasks} tasks with ≥1 success | "
          f"mean reward={sum(t.reward for t in all_trajs)/max(len(all_trajs),1):.3f} | "
          f"wall={wall/60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
