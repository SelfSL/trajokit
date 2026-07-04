"""Convert trajokit Trajectories into verl-style GRPO/GSPO batches.

verl conventions honored here:
- prompt  = initial prefix (the leading loss_mask==0 run: system + first user turn)
- response = everything after; loss_mask zeros tool/obs tokens inside it
- prompts are LEFT-padded, responses RIGHT-padded
- uid = task_id (the GRPO/GSPO group key for group-normalized advantages)
- reward: one scalar per trajectory; trainers typically place it at the last
  valid response position (index provided as `last_response_idx`)

Pure-python (no torch) so it is unit-testable anywhere; `to_torch_batch` does a
lazy torch import for the final tensor step.
"""
from __future__ import annotations

from typing import Any

from ..types import Trajectory


def split_prompt_response(traj: Trajectory) -> tuple[list[int], list[int], list[int]]:
    """Prompt ends where the first assistant (mask==1) token begins."""
    try:
        first = traj.loss_mask.index(1)
    except ValueError:  # no assistant tokens at all -> everything is prompt
        first = len(traj.loss_mask)
    return traj.input_ids[:first], traj.input_ids[first:], traj.loss_mask[first:]


def pad_batch(
    groups: dict[str, list[Trajectory]],
    pad_id: int,
    max_prompt_len: int | None = None,
    max_response_len: int | None = None,
) -> dict[str, Any]:
    """Flatten GRPO groups into aligned, padded lists (verl layout).

    Returns dict of equal-length lists:
      prompt_ids, prompt_attn (left-padded), response_ids, response_attn,
      response_loss_mask (right-padded), uid, reward, last_response_idx
    """
    trajs = [(uid, t) for uid, group in groups.items() for t in group]
    if not trajs:
        raise ValueError("empty batch")

    split = [(uid, *split_prompt_response(t), t.reward) for uid, t in trajs]
    p_len = max_prompt_len or max(len(p) for _, p, _, _, _ in split)
    r_len = max_response_len or max(len(r) for _, _, r, _, _ in split)

    out: dict[str, list] = {k: [] for k in (
        "prompt_ids", "prompt_attn", "response_ids", "response_attn",
        "response_loss_mask", "uid", "reward", "last_response_idx")}

    for uid, prompt, resp, rmask, reward in split:
        prompt = prompt[-p_len:]                      # clip long prompts from the LEFT
        resp, rmask = resp[:r_len], rmask[:r_len]     # clip long responses from the RIGHT
        pad_p = p_len - len(prompt)
        pad_r = r_len - len(resp)
        out["prompt_ids"].append([pad_id] * pad_p + prompt)
        out["prompt_attn"].append([0] * pad_p + [1] * len(prompt))
        out["response_ids"].append(resp + [pad_id] * pad_r)
        out["response_attn"].append([1] * len(resp) + [0] * pad_r)
        out["response_loss_mask"].append(rmask + [0] * pad_r)
        out["uid"].append(uid)
        out["reward"].append(reward)
        out["last_response_idx"].append(max(len(resp) - 1, 0))
    return out


def to_torch_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Lists -> torch tensors (uid stays a list). Lazy torch import."""
    import torch  # optional dep

    t = lambda k, dt: torch.tensor(batch[k], dtype=dt)  # noqa: E731
    input_ids = torch.cat([t("prompt_ids", torch.long), t("response_ids", torch.long)], dim=1)
    attention_mask = torch.cat([t("prompt_attn", torch.long), t("response_attn", torch.long)], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": t("response_ids", torch.long),
        "response_mask": t("response_loss_mask", torch.long),
        "rewards": t("reward", torch.float),
        "last_response_idx": t("last_response_idx", torch.long),
        "uid": batch["uid"],
    }
