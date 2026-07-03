import asyncio
from transformers import AutoTokenizer
from trajokit import AgentLoop, PolicyClient, LocalDockerSandbox, load_tasks

async def main():
    task = load_tasks("tasks.jsonl")[0]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    loop = AgentLoop(tokenizer=tok, max_turns=15, max_context=32768,
                    chat_template_kwargs={"enable_thinking": False})
    policy = PolicyClient("http://localhost:8000", model="Qwen/Qwen3-8B")
    traj = await loop.run(task, policy, LocalDockerSandbox())
    print(f"reward={traj.reward} turns={traj.info['turns']} submitted={traj.info['submitted']}\n")
    print(tok.decode(traj.input_ids))

asyncio.run(main())
