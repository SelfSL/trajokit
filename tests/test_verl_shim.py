"""VerlPolicyShim: token round-trip against a fake server manager."""
from types import SimpleNamespace

from trajokit.adapters.verl_agent_loop import VerlPolicyShim


class FakeTok:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeServerManager:
    def __init__(self):
        self.calls = []

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.calls.append((request_id, list(prompt_ids), dict(sampling_params)))
        return SimpleNamespace(token_ids=[104, 105], stop_reason="completed")  # "hi"


async def test_shim_round_trip_and_params():
    mgr = FakeServerManager()
    shim = VerlPolicyShim(mgr, FakeTok(), request_id="r1",
                          base_sampling_params={"temperature": 0.7, "top_p": 0.9})
    out = await shim.complete([1, 2, 3], max_tokens=17, stop=["</s>"])
    assert out == {"text": "hi", "token_ids": [104, 105], "finish_reason": "completed"}
    rid, pids, sp = mgr.calls[0]
    assert rid == "r1" and pids == [1, 2, 3]
    assert sp["max_tokens"] == 17 and sp["temperature"] == 0.7 and sp["top_p"] == 0.9
    assert sp["stop"] == ["</s>"]


async def test_shim_sticky_request_id_across_turns():
    mgr = FakeServerManager()
    shim = VerlPolicyShim(mgr, FakeTok(), request_id="sticky")
    await shim.complete([1], max_tokens=1)
    await shim.complete([1, 2], max_tokens=1)
    assert [c[0] for c in mgr.calls] == ["sticky", "sticky"]  # same server -> prefix cache hits
