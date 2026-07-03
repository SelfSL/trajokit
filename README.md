# trajokit

Sandboxed agent rollouts for RL. Give it tasks and a policy endpoint; get back token
trajectories with correct loss masks.

```
Tasks (jsonl) → Orchestrator → AgentLoop ⇄ Policy (OpenAI-compatible HTTP)
                                   ⇅
                                Sandbox (docker)
                                   ↓
                    Trajectory {input_ids, loss_mask, reward}
```

**Why:** in agentic RL, the hardest silent bugs are token/mask misalignment from
retokenizing conversations, and reward poisoning from scoring infra failures as 0.
trajokit owns the loop so both are solved once: incremental append-only tokenization,
and env-failure ⇒ drop (never 0).

**Design:** everything is data (`Task`, `Trajectory`) or a swappable protocol
(`Sandbox`, policy endpoint, verifier). The only privileged code is the loop, and it
is tested to death.

## Quickstart

```python
import asyncio
from transformers import AutoTokenizer
from trajokit import AgentLoop, Orchestrator, PolicyClient, LocalDockerSandbox, load_tasks

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
loop = AgentLoop(tokenizer=tok, max_turns=30, max_context=32768)
policy = PolicyClient("http://localhost:8000", model="Qwen/Qwen3-8B")
orch = Orchestrator(loop, policy, sandbox_factory=LocalDockerSandbox, max_concurrency=64)

tasks = load_tasks("swebench_verified.jsonl")
groups = asyncio.run(orch.run_batch(tasks[:8], k=8))  # 8 tasks x 8 rollouts (GRPO groups)
```

## Status

v0.0.1 scaffold. Roadmap: SWE-bench Verified loader → verl adapter → recipe with
mid-scale numbers → τ²-bench-style customer-service env with judge-based rewards.

Apache-2.0
