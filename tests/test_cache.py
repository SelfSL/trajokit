"""RolloutCache: save/claim semantics."""
from trajokit.cache import RolloutCache
from trajokit.types import Trajectory


def _traj(tid="t1", reward=1.0):
    return Trajectory(task_id=tid, input_ids=[1, 2, 3], loss_mask=[0, 1, 1],
                      reward=reward, turn_spans=[(1, 3)], logprobs=[0.0, -0.5, -0.5])


def test_save_and_load_round_trip(tmp_path):
    c = RolloutCache(tmp_path, run_id="r1")
    c.save(_traj())
    out = c.load_one("t1")
    assert out is not None and out.input_ids == [1, 2, 3] and out.reward == 1.0
    assert out.turn_spans == [(1, 3)] and out.logprobs == [0.0, -0.5, -0.5]


def test_claims_prevent_double_handout_within_run(tmp_path):
    c = RolloutCache(tmp_path, run_id="r1")
    c.save(_traj()); c.save(_traj(reward=0.0))
    assert c.load_one("t1") is not None
    assert c.load_one("t1") is not None
    assert c.load_one("t1") is None  # both claimed


def test_new_run_id_reuses_cache(tmp_path):
    c1 = RolloutCache(tmp_path, run_id="r1")
    c1.save(_traj())
    assert c1.load_one("t1") is not None
    c2 = RolloutCache(tmp_path, run_id="r2")
    assert c2.load_one("t1") is not None  # fresh namespace


def test_miss_returns_none(tmp_path):
    assert RolloutCache(tmp_path).load_one("nope") is None

