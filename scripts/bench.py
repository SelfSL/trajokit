import asyncio, time
from transformers import AutoTokenizer
from trajokit import AgentLoop, Orchestrator, PolicyClient, LocalDockerSandbox, load_tasks

async def main():
    tasks = load_tasks("tasks.jsonl")[:5]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    loop = AgentLoop(tokenizer=tok, max_turns=15, max_context=32768,
                    chat_template_kwargs={"enable_thinking": False})
    policy = PolicyClient("http://localhost:8000", model="Qwen/Qwen3-8B")
    orch = Orchestrator(loop, policy, sandbox_factory=LocalDockerSandbox, max_concurrency=8)

    t0 = time.time()
    groups = await orch.run_batch(tasks, k=4)
    wall = time.time() - t0

    total = sum(len(g) for g in groups.values())
    for tid, g in groups.items():
        print(f"{tid}: rewards={[t.reward for t in g]} turns={[t.info['turns'] for t in g]}")
    print(f"\n{total}/20 rollouts survived | wall={wall/60:.1f} min | {wall/max(total,1):.0f}s/rollout effective")

asyncio.run(main())
