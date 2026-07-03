"""Judge clients: one interface, three backends.

- OpenAIChatJudge: any /v1/chat/completions endpoint (OpenAI, vLLM, sglang, GLM self-hosted)
- AnthropicJudge: api.anthropic.com /v1/messages
- BedrockJudge: AWS Bedrock converse API (boto3, sync call wrapped in a thread)
"""
from __future__ import annotations

import asyncio
import os
from typing import Protocol

import httpx


class JudgeClient(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


class OpenAIChatJudge:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(self, system: str, user: str) -> str:
        r = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0,
                "max_tokens": 512,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class AnthropicJudge:
    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str = "https://api.anthropic.com", timeout: float = 120.0):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(self, system: str, user: str) -> str:
        r = await self._client.post(
            f"{self.base_url}/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": 512,
                "temperature": 0,
            },
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])


class BedrockJudge:
    """AWS Bedrock converse API. Requires `pip install trajokit[bedrock]`; auth via IAM role/env."""

    def __init__(self, model_id: str, region: str = "us-west-2"):
        self.model_id = model_id
        self.region = region
        self._client = None

    def _sync_converse(self, system: str, user: str) -> str:
        import boto3  # lazy: optional dep
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        resp = self._client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0},
        )
        return "".join(c.get("text", "") for c in resp["output"]["message"]["content"])

    async def complete(self, system: str, user: str) -> str:
        return await asyncio.to_thread(self._sync_converse, system, user)
