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
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--no-thinking", action="store_true",
                    help="disable thinking mode via chat template (Qwen3.x family)")
    ap.add_argument("--tail", type=int, default=0,
                    help="print only the last N chars of the decoded trajectory (0 = all)")
    return ap.parse_args()


async def main(args):
    task = load_tasks(args.tasks)[args.task_idx]
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
    traj = await loop.run(task, policy, LocalDockerSandbox())
    print(f"reward={traj.reward} turns={traj.info['turns']} "
          f"submitted={traj.info['submitted']} tokens={len(traj.input_ids)}\n")
    text = tok.decode(traj.input_ids)
    print(text[-args.tail:] if args.tail else text)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
