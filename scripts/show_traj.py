import argparse
import asyncio

from transformers import AutoTokenizer
from trajokit import AgentLoop, PolicyClient, LocalDockerSandbox, load_tasks


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                    help="served model name")
    ap.add_argument("--tokenizer", default="/mnt/raid5/models/Qwen3-Coder-30B-A3B-Instruct",
                    help="tokenizer path (defaults to --model if empty)")
    ap.add_argument("--tasks", default="tasks.jsonl")
    ap.add_argument("--task-idx", type=int, default=0, help="which task to run")
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--tail", type=int, default=0,
                    help="print only the last N chars of the decoded trajectory (0 = all)")
    ap.add_argument("--temperature", type=float, default=0.2)
    return ap.parse_args()


async def main(args):
    task = load_tasks(args.tasks)[args.task_idx]
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    loop = AgentLoop(tokenizer=tok, max_turns=args.max_turns, max_context=args.max_context,
                 gen_kwargs={"temperature": args.temperature})
    policy = PolicyClient("http://localhost:8000", model=args.model)
    traj = await loop.run(task, policy, LocalDockerSandbox())
    print(f"reward={traj.reward} turns={traj.info['turns']} "
          f"submitted={traj.info['submitted']} tokens={len(traj.input_ids)}\n")
    text = tok.decode(traj.input_ids)
    print(text[-args.tail:] if args.tail else text)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
