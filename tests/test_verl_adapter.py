"""verl adapter: split/pad invariants (pure python, no torch needed)."""
import pytest

from trajokit.adapters.verl import pad_batch, split_prompt_response
from trajokit.types import Trajectory


def _traj(uid, n_prompt, spans, reward=0.0):
    """spans = list of (n_assistant, n_obs) turns."""
    ids, mask = list(range(n_prompt)), [0] * n_prompt
    for n_a, n_o in spans:
        ids += [1] * n_a + [2] * n_o
        mask += [1] * n_a + [0] * n_o
    return Trajectory(task_id=uid, input_ids=ids, loss_mask=mask, reward=reward)


def test_split_prompt_response():
    t = _traj("a", 5, [(3, 2), (4, 0)])
    p, r, m = split_prompt_response(t)
    assert len(p) == 5 and len(r) == 9 and len(m) == 9
    assert sum(m) == 7  # only assistant tokens trainable


def test_split_all_prompt():
    t = _traj("a", 4, [])
    p, r, m = split_prompt_response(t)
    assert len(p) == 4 and r == [] and m == []


def test_pad_batch_shapes_and_padding_sides():
    groups = {"t1": [_traj("t1", 3, [(2, 1)], 1.0), _traj("t1", 5, [(4, 2), (1, 0)], 0.0)]}
    b = pad_batch(groups, pad_id=9)
    # aligned lengths
    assert {len(x) for x in b["prompt_ids"]} == {5}
    assert {len(x) for x in b["response_ids"]} == {7}
    # prompts left-padded, responses right-padded
    assert b["prompt_ids"][0][:2] == [9, 9] and b["prompt_attn"][0][:2] == [0, 0]
    assert b["response_ids"][0][-4:] == [9, 9, 9, 9] and b["response_loss_mask"][0][-4:] == [0] * 4
    # grouping + rewards + last idx
    assert b["uid"] == ["t1", "t1"]
    assert b["reward"] == [1.0, 0.0]
    assert b["last_response_idx"] == [2, 6]


def test_pad_batch_clipping():
    groups = {"t": [_traj("t", 10, [(10, 5)])]}
    b = pad_batch(groups, pad_id=0, max_prompt_len=4, max_response_len=6)
    assert len(b["prompt_ids"][0]) == 4 and len(b["response_ids"][0]) == 6
    assert sum(b["response_loss_mask"][0]) == 6  # right-clip keeps leading assistant tokens


def test_empty_batch_raises():
    with pytest.raises(ValueError):
        pad_batch({}, pad_id=0)
