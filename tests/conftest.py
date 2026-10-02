from __future__ import annotations

from typing import Any

import pytest

from furrow.config import Provider, Settings


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"provider": Provider.OLLAMA}
    base.update(overrides)
    return Settings(**base)


class FakeClient:
    """Minimal stand-in for LLMClient used across the test-suite."""

    def __init__(
        self,
        response: str = "",
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings if settings is not None else make_settings()
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        return self.response

    async def read_file(self, path: str) -> str:
        return self.response

    async def write_file(self, path: str, content: str) -> None:
        pass

    def list_files(self, directory: str) -> list[str]:
        return []


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()
