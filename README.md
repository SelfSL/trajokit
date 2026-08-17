# trajokit

Sandboxed agent rollouts for RL post-training. Tasks in → token trajectories with
correct loss masks out.

```
Tasks (jsonl) → Orchestrator → AgentLoop ⇄ Policy (OpenAI-compatible HTTP / trainer server)
                                   ⇅
                                Sandbox (docker)
                                   ↓
                 Trajectory {input_ids, loss_mask, reward, logprobs}
                                   ↓
                     trainer adapter (verl shipped; slime/miles planned)
```

## Core contribution

The hardest bugs in agentic RL are silent: mask/token misalignment from
retokenizing conversations, and reward poisoning from scoring infra failures as 0.
trajokit is the **rollout layer alone, with correctness as the product**:

- **Token-id round-trip.** Prompts go to the engine as token ids, completions come
  back as token ids, the buffer is append-only, no chat-template re-rendering,
  ever. Verified in training: rollout↔trainer logprob Pearson 0.997.
- **Template-delta correctness.** Turn seams are built by single-render sentinel
  splicing, surviving position-polymorphic chat templates (e.g. Qwen3.x thinking
  blocks). Fixing one silent seam bug moved Qwen3.6-27B from 33% to 57% pass@1 on
  SWE-bench Verified-100, larger than any prompt change we tested.
- **Trainer-agnostic contract.** Output is just `{input_ids, loss_mask, reward,
  logprobs}` over an HTTP or token-server policy. The same episode code produced
  our standalone SWE-bench baselines and a 15-step GSPO run inside verl.
- **Reward hygiene.** Env failure drops the rollout (never a fake 0), verifiers are
  pluggable (unit tests or LLM judge), prompts are versioned with graded numbers.

Plus the operational pieces recipes need: rollout cache for instant replay
debugging, patch export for official-harness grading, SWE-bench Verified loader.

## How this composes with verl / RL-Factory / AgentRL

Frameworks like verl, RL-Factory, and AgentRL own the **trainer**: distributed
optimization, weight sync, rollout scheduling, and their own built-in agentic
loops. trajokit deliberately does not compete there, it owns the **trajectory
contract**: the agent loop, tokenization/masking, sandbox execution, and reward
verification, emitting trainer-ready tensors. Use it *with* them: our shipped verl
integration is ~100 lines mapping `Trajectory → AgentLoopOutput` over verl's token
server, and the same trajectory shape maps onto slime/miles `Sample`. If you're
happy with your trainer's native loop, keep it; reach for trajokit when you want
the same audited episode code to run standalone (evals, data collection) and
inside whichever trainer you use next.

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
groups = asyncio.run(orch.run_batch(tasks[:8], k=8))  # GRPO groups
```

## Validation

Controls (positive/negative/verifier-agreement), officially graded SWE-bench
Verified baselines (Qwen3.6-27B: **62.0% pass@1 on the full 500** with a minimal
bash scaffold, greedy, non-thinking; Qwen3-Coder-30B-A3B: 19.8%), a 15-step GSPO
training run through verl, and a case study where fixing silent template-seam
corruption moved a model 33% to 57%: see
[docs/validation.md](docs/validation.md).
Install sharp edges (Blackwell/CUDA 13): [docs/install.md](docs/install.md).

## Roadmap

Prompt/format presets (`ActionFormat`, incl.
thinking-mode configs) → multi-GPU scaling recipes → τ²-bench-style
customer-service env with judge rewards → slime/miles adapters.

Apache-2.0
