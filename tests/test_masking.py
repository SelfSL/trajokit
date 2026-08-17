"""The tests that matter most: mask/token alignment invariants.

Uses a fake tokenizer + fake policy/sandbox so CI needs no GPUs, no docker, no HF.
"""
import pytest

from trajokit.loop import AgentLoop
from trajokit.types import ExecResult, Task, Trajectory


class FakeTokenizer:
    """Char-level tokenizer: 1 char = 1 token. Deterministic and drift-free."""

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=False):
        s = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in msgs)
        if add_generation_prompt:
            s += "<assistant>"
        return [ord(c) for c in s] if tokenize else s

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakePolicy:
    def __init__(self, scripted: list[str]):
        self.scripted = list(scripted)

    async def complete(self, prompt_token_ids, max_tokens=2048, **kw):
        text = self.scripted.pop(0)
        ids = [ord(c) for c in text]
        return {"text": text, "token_ids": ids, "logprobs": [-0.5] * len(ids), "finish_reason": "stop"}


class FakeSandbox:
    def __init__(self, pass_tests: bool = True):
        self.pass_tests = pass_tests

    async def start(self, env_spec):
        pass

    async def exec(self, cmd, timeout=120.0):
        if "runtests" in cmd:
            return ExecResult("OK", "", 0 if self.pass_tests else 1)
        return ExecResult("file contents here", "", 0)

    async def stop(self):
        pass


TASK = Task(task_id="t1", prompt="fix the bug", env_spec={"test_cmd": "runtests"})


@pytest.fixture
def loop():
    return AgentLoop(tokenizer=FakeTokenizer(), max_turns=5, max_context=100_000)


async def test_mask_alignment_and_reward(loop):
    policy = FakePolicy(["look:\n```bash\ncat f.py\n```", "done\nsubmit"])
    traj = await loop.run(TASK, policy, FakeSandbox(pass_tests=True))
    assert isinstance(traj, Trajectory)
    assert len(traj.input_ids) == len(traj.loss_mask)
    assert traj.reward == 1.0
    assert traj.info["submitted"] is True
    assert traj.info["turns"] == 2


async def test_masked_tokens_are_exactly_assistant_tokens(loop):
    a1, a2 = "x:\n```bash\ncat f.py\n```", "submit"
    policy = FakePolicy([a1, a2])
    traj = await loop.run(TASK, policy, FakeSandbox())
    n_masked = sum(traj.loss_mask)
    assert n_masked == len(a1) + len(a2)  # char-level fake: 1 char = 1 token
    # masked ids must decode back to exactly the assistant text
    masked = "".join(chr(i) for i, m in zip(traj.input_ids, traj.loss_mask) if m)
    assert masked == a1 + a2


async def test_failed_tests_give_zero_reward(loop):
    policy = FakePolicy(["```bash\ncat f.py\n```", "submit"])
    traj = await loop.run(TASK, policy, FakeSandbox(pass_tests=False))
    assert traj.reward == 0.0


async def test_max_turns_truncation(loop):
    policy = FakePolicy(["```bash\ncat f.py\n```"] * 5)
    traj = await loop.run(TASK, policy, FakeSandbox())
    assert traj.info["turns"] == 5
    assert traj.info["submitted"] is False


async def test_trajectory_length_invariant():
    with pytest.raises(ValueError):
        Trajectory(task_id="x", input_ids=[1, 2], loss_mask=[1], reward=0.0)


# ---- verifier tests ----

class FakeJudge:
    def __init__(self, reply: str):
        self.reply = reply
        self.last_user = None

    async def complete(self, system, user):
        self.last_user = user
        return self.reply


async def test_judge_verifier_parses_score():
    from trajokit.verifiers import JudgeVerifier
    judge = FakeJudge('Here you go: {"score": 0.75}')
    loop = AgentLoop(tokenizer=FakeTokenizer(), max_turns=5,
                     max_context=100_000, verifier=JudgeVerifier(judge))
    policy = FakePolicy(["```bash\ncat f.py\n```", "submit"])
    traj = await loop.run(TASK, policy, FakeSandbox())
    assert traj.reward == 0.75
    assert "[agent]" in judge.last_user and "[env]" in judge.last_user


async def test_judge_verifier_unparseable_raises():
    from trajokit.verifiers import JudgeVerifier
    loop = AgentLoop(tokenizer=FakeTokenizer(), max_turns=5,
                     max_context=100_000, verifier=JudgeVerifier(FakeJudge("great job!")))
    policy = FakePolicy(["submit"])
    with pytest.raises(ValueError):
        await loop.run(TASK, policy, FakeSandbox())


async def test_turn_spans_cover_masked_tokens(loop):
    policy = FakePolicy(["```bash\ncat f.py\n```", "submit"])
    traj = await loop.run(TASK, policy, FakeSandbox())
    in_span = set()
    for s, e in traj.turn_spans:
        in_span.update(range(s, e))
    masked = {i for i, m in enumerate(traj.loss_mask) if m}
    assert in_span == masked


async def test_no_duplicate_assistant_close():
    """Close fragment is present once, and eos-stripped when the model emitted it."""
    tok = FakeTokenizer()
    tok.eos_token_id = ord("<")  # decode -> "<", the first char of "</assistant>"
    loop2 = AgentLoop(tokenizer=tok, max_turns=3, max_context=100_000)
    with_close = tok.decode(loop2._obs_ids("obs", drop_assistant_close=False))
    without = tok.decode(loop2._obs_ids("obs", drop_assistant_close=True))
    assert with_close == "</assistant><user>obs</user><assistant>"
    assert without == with_close[1:]  # leading eos char stripped


class FakeThinkTokenizer(FakeTokenizer):
    """Mimics Qwen3.x thinking-disabled: generation prompt carries a suffix."""

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=False, **kw):
        s = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in msgs)
        if add_generation_prompt:
            s += "<assistant><T></T>"  # think suffix AFTER the assistant open
        return [ord(c) for c in s] if tokenize else s


async def test_obs_delta_survives_generation_prompt_suffix():
    """Regression: len()-based slicing corrupted seams when base isn't a prefix of full."""
    tok = FakeThinkTokenizer()
    loop2 = AgentLoop(tokenizer=tok, max_turns=3, max_context=100_000)
    delta = tok.decode(loop2._obs_ids("obs"))
    assert delta == "</assistant><user>obs</user><assistant><T></T>"


class FakePolymorphicTokenizer(FakeTokenizer):
    """Mimics Qwen3.6: FINAL assistant turn wrapped in think tags, history turns stripped."""

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=False, **kw):
        parts = []
        for i, m in enumerate(msgs):
            c = m["content"]
            if m["role"] == "assistant" and i == len(msgs) - 1 and not add_generation_prompt:
                c = f"<T>{c}</T>"  # final-position rendering differs from history
            parts.append(f"<{m['role']}>{c}</{m['role']}>")
        s = "".join(parts)
        if add_generation_prompt:
            s += "<assistant><T></T>"
        return [ord(c) for c in s] if tokenize else s


async def test_obs_delta_survives_position_polymorphic_template():
    """Regression: render-diff approaches break when history turns render differently."""
    tok = FakePolymorphicTokenizer()
    loop2 = AgentLoop(tokenizer=tok, max_turns=3, max_context=100_000)
    delta = tok.decode(loop2._obs_ids("obs"))
    assert delta == "</assistant><user>obs</user><assistant><T></T>"


async def test_unterminated_bash_fence_is_executed(loop):
    """Models often end the turn (eos) without closing the fence - accept it."""
    policy = FakePolicy(["```bash\ncat f.py", "submit"])
    traj = await loop.run(TASK, policy, FakeSandbox())
    assert traj.info["turns"] == 2 and traj.info["submitted"] is True


async def test_empty_bash_fence_gets_nudge(loop):
    policy = FakePolicy(["```bash\n", "```bash\ncat f.py\n```", "submit"])
    traj = await loop.run(TASK, policy, FakeSandbox())
    assert traj.info["turns"] == 3  # empty fence consumed a turn with a nudge, then real work


async def test_submit_inside_fence_does_not_end_episode(loop):
    """'submit' on its own line INSIDE a bash fence must not terminate the episode."""
    policy = FakePolicy(["```bash\necho hi\nsubmit\n```", "submit"])
    traj = await loop.run(TASK, policy, FakeSandbox())
    assert traj.info["turns"] == 2          # fence-submit ignored, real submit on turn 2
    assert traj.info["submitted"] is True


async def test_patch_captured_when_workdir_set(loop):
    task = Task(task_id="t2", prompt="fix", env_spec={"test_cmd": "runtests", "workdir": "/repo"})
    policy = FakePolicy(["```bash\ncat f.py\n```", "submit"])
    traj = await loop.run(task, policy, FakeSandbox())
    assert traj.info["patch"] == "file contents here"  # FakeSandbox generic exec output


async def test_logprobs_aligned_with_tokens(loop):
    policy = FakePolicy(["```bash\ncat f.py\n```", "submit"])
    traj = await loop.run(TASK, policy, FakeSandbox())
    assert traj.logprobs is not None and len(traj.logprobs) == len(traj.input_ids)
    # mask==0 positions carry 0.0 filler
    assert all(lp == 0.0 for lp, m in zip(traj.logprobs, traj.loss_mask) if m == 0)



async def test_fence_containing_only_submit_ends_episode(loop):
    """```bash\nsubmit\n``` is an unambiguous submit, not a command to execute."""
    policy = FakePolicy(["```bash\ncat f.py\n```", "```bash\nsubmit\n```"])
    traj = await loop.run(TASK, policy, FakeSandbox())
    assert traj.info["turns"] == 2 and traj.info["submitted"] is True
