import asyncio
from trajokit import LocalDockerSandbox, load_tasks
from trajokit.verifiers import TestCmdVerifier

async def main():
    task = load_tasks("tasks.jsonl")[0]
    sb = LocalDockerSandbox()
    await sb.start(task.env_spec)
    r = await TestCmdVerifier().score(task, sb, "")   # no agent edits
    await sb.stop()
    print("no-op reward:", r)

asyncio.run(main())
