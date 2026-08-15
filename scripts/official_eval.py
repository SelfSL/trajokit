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
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--no-thinking", action="store_true",
                    help="disable thinking mode via chat template (Qwen3.x family)")
    ap.add_argument("--out", default="predictions.jsonl")
    return ap.parse_args()


async def main(args):
    tasks = load_tasks(args.tasks)[: args.n_tasks]
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    gen_kwargs = {"temperature": args.temperature}
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    if args.presence_penalty is not None:
        gen_kwargs["presence_penalty"] = args.presence_penalty
    loop = AgentLoop(tokenizer=tok, max_turns=args.max_turns, max_context=args.max_context,
                     gen_kwargs=gen_kwargs,
                     chat_template_kwargs={"enable_thinking": False} if args.no_thinking else None)
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
          f"--predictions_path {args.out} --max_workers 12 --run_id <RUN_ID>")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
