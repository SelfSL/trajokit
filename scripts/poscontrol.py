import asyncio
from datasets import load_dataset
from trajokit import LocalDockerSandbox, load_tasks
from trajokit.verifiers import TestCmdVerifier

async def main():
    task = load_tasks("tasks.jsonl")[0]
    row = next(r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
               if r["instance_id"] == task.task_id)
    sb = LocalDockerSandbox()
    await sb.start(task.env_spec)
    res = await sb.exec("cd /testbed && git apply -v - <<'EOF'\n" + row["patch"] + "\nEOF\n")
    print("gold apply exit:", res.returncode)
    r = await TestCmdVerifier().score(task, sb, "")
    await sb.stop()
    print("gold reward:", r)   # expect 1.0

asyncio.run(main())
