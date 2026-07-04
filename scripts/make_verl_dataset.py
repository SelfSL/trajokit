"""tasks.jsonl -> verl parquet for the trajokit agent loop.

Columns:
  prompt      chat-format messages (verl uses this for length filtering/logging)
  agent_name  "trajokit" -> routes rows to TrajokitAgentLoop
  task_id, env_spec (JSON string)  passed through to the loop as kwargs
"""
import argparse
import json

import pandas as pd

from trajokit import load_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks_100.jsonl")
    ap.add_argument("--out", default="swebench_verl.parquet")
    ap.add_argument("--data-source", default="swebench_verified")
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    df = pd.DataFrame(
        {
            "prompt": [[{"role": "user", "content": t.prompt}] for t in tasks],
            "agent_name": "trajokit",
            "task_id": [t.task_id for t in tasks],
            "env_spec": [json.dumps(t.env_spec) for t in tasks],
            "data_source": args.data_source,
        }
    )
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
