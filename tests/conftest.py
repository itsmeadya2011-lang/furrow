"""Shared fixtures for Furrow tests."""
from __future__ import annotations

import pytest

from furrow.config import Provider, Settings


class FakeLLMClient:
    """A fake LLMClient that returns predetermined responses for testing."""

    def __init__(
        self,
        settings: Settings | None = None,
        response: str = "",
    ) -> None:
        self.settings = settings or Settings()
        self._response = response
        self.complete_calls: list[tuple[str, str, Optional[str]]] = []
        self.read_results: dict[str, str] = {}
        self.write_calls: list[tuple[str, str]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls.append((prompt, system, model))
        return self._response

    async def read_file(self, path: str) -> str:
        return self.read_results.get(path, "")

    async def write_file(self, path: str, content: str) -> None:
        self.write_calls.append((path, content))

    def list_files(self, directory: str = ".") -> list[str]:
        return []


@pytest.fixture
def fake_settings() -> Settings:
    """Return test settings with a dummy provider."""
    return Settings(
        provider=Provider.ANTHROPIC,
        anthropic_api_key="test-key",
        openai_api_key="test-key",
        max_parallel_tasks=5,
        max_cycles=0,
    )


@pytest.fixture
def fake_client(fake_settings: Settings) -> FakeLLMClient:
    """Return a FakeLLMClient with test settings."""
    return FakeLLMClient(settings=fake_settings)
