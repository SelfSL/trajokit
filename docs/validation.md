# Validation

Hardware for all results: 1× RTX Pro 6000 (vLLM serving), local Docker sandboxes,
minimal bash-agent scaffold.

## Correctness controls
- **Unit tests** (`tests/`): 28 passing, mask/token alignment invariants,
  template-delta seam regressions (position-polymorphic templates, generation-prompt
  suffixes), fence parsing (fenced submit, unterminated fences, empty fences),
  duplicate-eos, logprob alignment, patch capture, rollout cache claims, verl
  adapter padding/splitting. No GPU/docker needed.
- **Negative control** (`scripts/negcontrol.py`): untouched repo → reward 0.0.
- **Positive control** (`scripts/poscontrol.py`): gold patch applied → reward 1.0.
- **Verifier agreement**: in-container reward matched official-harness verdicts on
  the 100-task evals below.
- **Training-side verification**: rollout↔trainer logprob Pearson 0.997 in the verl
  GSPO run (token-id round-trip holds end to end).

## End-to-end runs
- `scripts/show_traj.py`: single-episode inspection; first agent-earned reward=1.0
  on django__django-11099 (correct \A...\Z regex fix).
- `scripts/bench.py`: k=4 GRPO-style groups. Django slice (10 tasks, temp 0.2,
  30 turns): 3/10 tasks with ≥1 success, mean reward 0.225, mixed groups
  (nonzero GRPO gradient) confirmed.
- **verl training**: 15-step GSPO run (Qwen3-8B, FSDP2 offload, single GPU)
  completed with rewards/mean 0.0625, grad_norm 0.35, staleness tags 0.

## Official SWE-bench Verified results

All runs: trajokit bash-minimal scaffold, greedy (temp 0), k=1 attempt, 50 turns,
32k episode context, official per-instance Docker images, graded by
`swebench.harness.run_evaluation`. Qwen3.x-family models run non-thinking
(`enable_thinking: False`).

### Model comparison, first-100 slice (astropy+django)

| model | config notes | pass@1 | empty patches |
|---|---|---|---|
| Qwen3.6-27B (dense) | non-thinking; post seam-fix | **57/100** | 12 |
| Qwen3-Coder-30B-A3B (MoE) | bash-minimal-v1; flashinfer absent | 24/100 | 24 |
| Qwen3-Coder-30B-A3B (MoE) | v1 + multi-block note; flashinfer sm_120 source build | 16/100 | 41 |

Open confound: all coder runs scoring 16-17% used the source-built sm_120
flashinfer; the 24% run predates it. Discriminating rerun (no flashinfer, current
code) pending.

### Full 500

| model | pass@1 | empty patches |
|---|---|---|
| Qwen3.6-27B (dense, non-thinking, post seam-fix) | **310/500 (62.0%)** | 50 (10%) |
| Qwen3-Coder-30B-A3B (MoE, bash-minimal-v1) | 99/500 (19.8%) | 190 (38%) |

Context for the 62.0%: on the same model, the community reports 67.8% via
mini-swe-agent (richer settings), 77.2% via Qwen's official scaffold, and up to
90% with an engineered agent stack. Our minimal-scaffold greedy non-thinking
number sits below all of these, exactly where it should: the scaffold ladder is
consistent, so the number is credible rather than inflated.

Coder empty-patch rate rises from 24% (first-100 slice) to 38% on the full set:
the minimal scaffold generalizes worse to unseen repos for that model.

## Case study: template-delta seam corruption (fixed)

With `enable_thinking: False`, Qwen3.x chat templates render assistant turns
*position-dependently*: the final turn keeps the `<think>` block, history turns
strip it. Our observation-delta previously assumed one template render is a string
prefix of another; the slice offset landed mid-token and corrupted every turn seam
(`<|im_end|>rt|>user`). Conditioned on garbage, the model degenerated into empty
bash fences, 50/100 empty patches, 33/100 resolved.

Fix: single-render sentinel splicing (`AgentLoop._obs_frag`), one dummy render cut
at literal sentinel positions, no cross-render assumptions, plus acceptance of
end-of-message-terminated bash fences and a nudge for empty fences.

Result, same model/slice/config: **33% to 57% pass@1, empty patches 50 → 12** -
a +24-point swing from token-stream correctness alone, larger than any prompt or
model change we tested. Regression tests:
`test_obs_delta_survives_position_polymorphic_template`,
`test_obs_delta_survives_generation_prompt_suffix`,
`test_unterminated_bash_fence_is_executed`.

Lesson encoded: token-stream corruption is silent and dwarfs prompt effects;
transcript-level auditing is part of the eval loop, not optional.

## cline-bench (first non-SWE-bench domain)

Adapter: `trajokit.datasets.clinebench` maps a local clone of
github.com/cline/cline-bench (Harbor task format) onto the standard Task shape.
Per-task images are built locally from each task's Dockerfile; tests are injected
at verification time only (base64 tarball inside test_cmd), so the agent never
sees them; grading gates on the upstream reward.txt convention; the sandbox gets
per-task network (`bridge`) because the upstream verifier and oracle require
egress by design.

Controls (task 0): negative = 7 genuine test failures on the untouched repo
(exit 1); positive = oracle solve.sh then 7/7 pass (exit 0).

| model | config | solved |
|---|---|---|
| Qwen3.6-27B (non-thinking, greedy, 50 turns, 32k) | bash-minimal | **3/11 (27%)** |
| Qwen3.8-27B (thinking default, greedy, 50 turns, 32k) | bash-minimal | 2/11 (18%) |
| Qwen3.8-27B (thinking default, greedy, 50 turns, 128k) | bash-minimal | 2/11 (18%) |

The 128k row is a one-variable experiment (episode budget 32k to 128k, all else
fixed): identical score, same two solves, 4x wall-clock, and 10/11 episodes ran
to the 50-turn cap. Context was not the binding constraint for Qwen3.8 here;
transcript autopsies show the failure mode is behavioral: reconnaissance
exhaustion (20+ read-only turns) followed by writing the full solution as a
typescript-fenced plan instead of executable edit commands, so no file is ever
modified. 32k remains the default config.

Config-hygiene lesson from this experiment: the episode budget and the server's
max_model_len must be raised together; a stale server silently truncates prompts
server-side. `scripts/serve_and_bench.sh` now hard-verifies the served config via
the API before launching.

Environment coverage: 11/12 upstream tasks build; 1 skipped due to an upstream
Dockerfile breakage (unpinned torch + torch-scatter sdist with
--no-build-isolation fails at metadata generation; deterministic, reported
upstream once their issue tracker opens).

Caveats carried on this benchmark: n=11 (noisy), public-repo provenance
(contamination risk), locally built environments (dependency drift between
builds, unlike SWE-bench's frozen images).

## Prompt versioning

The system prompt is part of the recipe; changes are only adopted with a graded eval.

| prompt | change | pass@1 (first 100 Verified, 30B coder) | empty patches |
|---|---|---|---|
| bash-minimal-v1 | baseline | **24/100** | 24 |
| bash-minimal-v2 (rejected) | added "graded only on final repo state; explanations score zero; verify before submit" | 17/100 | 42 |
| bash-minimal-v3 (rejected) | affirmative writable-checkout framing + multi-block feedback (two variables, flawed experiment design) | 16/100 | 43 |

v2 failed: the framing induced early bail-outs and "analysis complete" loops where
the model documented the fix, never applied it, and repeated `exit 0`. v3 bundled
two changes and regressed; only the mechanical multi-block feedback note was kept.
Reports live in `docs/results/`.

## Bugs found by these controls (fixed)
1. Heredoc terminator swallowed the test command → verifier always 1.0 (caught by negcontrol)
2. Duplicate `<|im_end|>` per turn → token-stream corruption (caught by transcript inspection)
3. `submit` inside a bash fence ended episodes early (caught by transcript inspection)
4. Django test-ID format broke the verifier → false negatives (caught by poscontrol on django)
5. Template-delta seam corruption under position-polymorphic templates → degenerate
   episodes (caught by transcript inspection + seam probe; see case study)
