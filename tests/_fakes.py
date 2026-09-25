from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiofiles  # type: ignore[import-untyped]

from furrow.config import Provider, Settings


class FakeClient:
    """Minimal stand-in for LLMClient used in tests."""

    def __init__(
        self,
        settings: Settings,
        responses: list[str] | None = None,
        complete_fn: Any | None = None,
        delay: float = 0.0,
    ) -> None:
        self.settings = settings
        self.responses = list(responses or [])
        self._idx = 0
        self.complete_fn = complete_fn
        self.delay = delay
        self.concurrent = 0
        self.max_concurrent = 0

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.complete_fn is not None:
                return await self.complete_fn(prompt, system, model)
            if self._idx < len(self.responses):
                resp = self.responses[self._idx]
                self._idx += 1
                return resp
            return ""
        finally:
            self.concurrent -= 1

    async def read_file(self, path: str | Path) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = self.settings.workspace / p
        async with aiofiles.open(p) as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        if not p.is_absolute():
            p = self.settings.workspace / p
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path = ".") -> list[str]:
        p = self.settings.workspace / directory
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]


class CollectingReporter:
    """Captures printed messages without touching stdout/stderr."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, message: Any) -> None:
        self.messages.append(str(message))


def make_settings(workspace: Path, **overrides: Any) -> Settings:
    kwargs: dict[str, Any] = {
        "provider": Provider.ANTHROPIC,
        "model": "x",
        "planner_model": "x",
        "worker_model": "x",
        "tester_model": "x",
        "anthropic_api_key": "sk-fake",
        "openai_api_key": None,
        "ollama_base_url": "http://localhost:11434",
        "max_parallel_tasks": 5,
        "max_cycles": 0,
        "log_level": "WARNING",
        "workspace": workspace,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)
