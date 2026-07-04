import argparse, asyncio
from trajokit import LocalDockerSandbox, load_tasks

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks_100.jsonl")
    ap.add_argument("--task-idx", type=int, default=0)
    return ap.parse_args()

async def main(a):
    t = load_tasks(a.tasks)[a.task_idx]
    sb = LocalDockerSandbox()
    await sb.start(t.env_spec)
    r = await sb.exec(t.env_spec["test_cmd"], timeout=900)
    print("exit:", r.returncode)
    print("STDOUT tail:\n", r.stdout[-2000:])
    print("STDERR tail:\n", r.stderr[-1200:])
    await sb.stop()

asyncio.run(main(parse_args()))
