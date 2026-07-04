"""Policy = any OpenAI-compatible /v1/completions endpoint (vLLM/sglang server mode).

We use the *completions* (not chat) endpoint on purpose: the loop owns chat-template
rendering so that tokenization is controlled in exactly one place.
"""
from __future__ import annotations

from typing import Any

import httpx


class PolicyClient:
    def __init__(self, base_url: str, model: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(
        self,
        prompt_token_ids: list[int],
        max_tokens: int = 2048,
        temperature: float = 1.0,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send token ids, get text + token ids back.

        vLLM accepts token-id prompts and can return token ids
        (`return_tokens_as_token_ids`), which removes retokenization drift entirely.
        """
        payload = {
            "model": self.model,
            "prompt": prompt_token_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop or [],
            "logprobs": 0,
            "return_tokens_as_token_ids": True,
        }
        r = await self._client.post(f"{self.base_url}/v1/completions", json=payload)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        token_ids = None
        lp = choice.get("logprobs") or {}
        if lp.get("tokens"):
            # vLLM formats as "token_id:12345" when return_tokens_as_token_ids=True
            token_ids = [int(t.split(":")[-1]) for t in lp["tokens"]]
        return {
            "text": choice["text"],
            "token_ids": token_ids,
            "logprobs": lp.get("token_logprobs"),
            "finish_reason": choice.get("finish_reason"),
        }

    async def aclose(self) -> None:
        await self._client.aclose()

