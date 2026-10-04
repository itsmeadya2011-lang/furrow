from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from furrow.config import Settings


class FakeLLMClient:
    """Minimal stand-in for :class:`furrow.llm.LLMClient`.

    Only exposes what the agents touch: a ``settings`` object and an async
    ``complete()`` that replays queued responses in order.
    """

    def __init__(self, responses: list[Any] | None = None, settings: Settings | None = None) -> None:
        self.responses: list[Any] = list(responses or [])
        self.settings = settings
        self.calls: list[dict] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        if not self.responses:
            raise AssertionError("FakeLLMClient has no queued response left")
        nxt = self.responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


@pytest.fixture
def settings_factory(tmp_path: Path):
    """Build ``Settings`` instances rooted in ``tmp_path`` (never the repo cwd)."""

    def _make(**overrides: Any) -> Settings:
        base: dict[str, Any] = {
            "state_file": tmp_path / "state.json",
            "workspace": tmp_path,
            "max_parallel_tasks": 5,
            "max_cycles": 0,
        }
        base.update(overrides)
        return Settings(**base)

    return _make


@pytest.fixture
def fake_llm_cls():
    return FakeLLMClient
