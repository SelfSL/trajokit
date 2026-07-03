"""Sandbox protocol + local Docker backend.

Backend #1 is the docker CLI via asyncio subprocess: zero deps, good enough for
hundreds of concurrent containers on one box. Remote pools implement the same
Protocol later without touching loop code.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Protocol

from .types import ExecResult


class Sandbox(Protocol):
    async def start(self, env_spec: dict[str, Any]) -> None: ...
    async def exec(self, cmd: str, timeout: float = 120.0) -> ExecResult: ...
    async def stop(self) -> None: ...


class LocalDockerSandbox:
    """One container per rollout. Locked down: no network, IMDS unreachable, capped cpu/mem."""

    def __init__(self, cpus: float = 2.0, mem: str = "8g", network: str = "none"):
        self.cpus, self.mem, self.network = cpus, mem, network
        self.name: str | None = None

    async def start(self, env_spec: dict[str, Any]) -> None:
        self.name = f"trajokit-{uuid.uuid4().hex[:12]}"
        self.workdir = env_spec.get("workdir", "/")
        cmd = [
            "docker", "run", "-d", "--name", self.name,
            f"--cpus={self.cpus}", f"--memory={self.mem}",
            f"--network={self.network}",
            "--security-opt", "no-new-privileges",
            env_spec["image"], "sleep", "infinity",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"container start failed: {err.decode()[-500:]}")

    async def exec(self, cmd: str, timeout: float = 120.0) -> ExecResult:
        assert self.name, "sandbox not started"
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-w", self.workdir, self.name, "bash", "-lc", cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return ExecResult(out.decode(errors="replace"), err.decode(errors="replace"), proc.returncode or 0)
        except asyncio.TimeoutError:
            proc.kill()
            return ExecResult("", f"command timed out after {timeout}s", -1, timed_out=True)

    async def stop(self) -> None:
        if self.name:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", self.name,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            self.name = None
