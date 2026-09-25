"""Shared test fixtures and mocks."""
from __future__ import annotations

import pytest
from furrow.config import Settings


class MockLLMClient:
    """Mock LLM client for testing without API calls."""

    def __init__(self, responses: list[str] | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._responses = responses or []
        self._call_index = 0
        self.call_log: list[dict] = []

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        self.call_log.append({"prompt": prompt, "model": model})
        if self._call_index < len(self._responses):
            response = self._responses[self._call_index]
            self._call_index += 1
            return response
        return "{}"

    async def read_file(self, path: str) -> str:
        return ""

    async def write_file(self, path: str, content: str) -> None:
        pass

    def list_files(self, directory: str) -> list[str]:
        return []


@pytest.fixture
def mock_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def mock_settings() -> Settings:
    return Settings()
