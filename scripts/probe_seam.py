"""Probe the exact seam construction for Qwen3.6 with thinking disabled.

Prints: eos identity, the extracted assistant-close fragment, template renders
around the splice point, both obs-delta variants, and whether the live server
includes the eos token in returned token ids.
"""
import asyncio
import sys

sys.path.insert(0, "src")

from transformers import AutoTokenizer

from trajokit.loop import AgentLoop
from trajokit.policy import PolicyClient
from trajokit.types import Task


def main_static():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    loop = AgentLoop(tokenizer=tok, chat_template_kwargs={"enable_thinking": False})
    ct = {"enable_thinking": False}

    print("eos_token_id:", tok.eos_token_id)
    print("eos decode  :", repr(tok.decode([tok.eos_token_id])))
    print("close repr  :", repr(loop._assistant_close()))

    mid = tok.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": ""}],
        tokenize=False, **ct,
    )
    full = tok.apply_chat_template(
        [{"role": "user", "content": "x"},
         {"role": "assistant", "content": ""},
         {"role": "user", "content": "OBS"}],
        add_generation_prompt=True, tokenize=False, **ct,
    )
    print("mid tail    :", repr(mid[-60:]))
    print("full@midend :", repr(full[max(len(mid) - 20, 0):len(mid) + 40]))
    print("lcp(mid,full):", loop._lcp(mid, full), "len(mid):", len(mid))
    print("delta drop=F:", repr(tok.decode(loop._obs_ids("OBS", drop_assistant_close=False))))
    print("delta drop=T:", repr(tok.decode(loop._obs_ids("OBS", drop_assistant_close=True))))
    return tok, loop


async def main_live(tok, loop):
    policy = PolicyClient("http://localhost:8000", model="Qwen/Qwen3.6-27B")
    task = Task(task_id="probe", prompt="Say hi in one word.", env_spec={})
    prefix = loop._prefix_ids(task)
    out = await policy.complete(prefix, max_tokens=32, temperature=0.0)
    ids = out["token_ids"] or []
    print("gen last 5 ids:", ids[-5:])
    print("ends with eos :", bool(ids) and ids[-1] == tok.eos_token_id)
    print("gen decode    :", repr(tok.decode(ids)))
    print("finish_reason :", out["finish_reason"])
    await policy.aclose()


if __name__ == "__main__":
    tok, loop = main_static()
    asyncio.run(main_live(tok, loop))
