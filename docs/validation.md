# Validation

All results: Qwen3-Coder-30B-A3B-Instruct served via vLLM on 1× RTX Pro 6000,
local Docker sandboxes, minimal bash-agent scaffold.

## Correctness controls
- **Unit tests** (`tests/`): 11 passing — mask/token alignment invariants,
  fence-submit parsing, duplicate-eos regression, patch capture. No GPU/docker needed.
- **Negative control** (`scripts/negcontrol.py`): untouched repo → reward 0.0.
- **Positive control** (`scripts/poscontrol.py`): gold patch applied → reward 1.0.
- **Verifier agreement**: in-container reward matched official-harness verdicts on
  the 100-task eval below.

## End-to-end runs
- `scripts/show_traj.py`: single-episode inspection; first agent-earned reward=1.0
  on django__django-11099 (correct \A...\Z regex fix).
- `scripts/bench.py`: k=4 GRPO-style groups. Django slice (10 tasks, temp 0.2,
  30 turns): 3/10 tasks with ≥1 success, mean reward 0.225, mixed groups
  (nonzero GRPO gradient) confirmed.

## Official SWE-bench Verified eval (first 100 instances)
`scripts/official_eval.py` (greedy, 50 turns, 32k ctx) → patch export →
`swebench.harness.run_evaluation`:

| metric | value |
|---|---|
| pass@1 (official harness) | **24/100** |
| empty patches | 24 |
| harness errors | 0 |

Known gaps: minimal scaffold (no edit tools), 32k context, obs truncation at 8k
chars. Published numbers for this model (~31%) use richer scaffolds + 128k ctx.

## Bugs found by these controls (fixed)
1. Heredoc terminator swallowed test command → verifier always 1.0 (caught by negcontrol)
2. Duplicate `<|im_end|>` per turn → token-stream corruption (caught by transcript inspection)
3. `submit` inside a bash fence ended episodes early (caught by transcript inspection)
4. Django test-ID format broke the verifier → false negatives (caught by poscontrol on django)
