from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TestResult, settings

# Prevent pytest from trying to collect framework classes whose names start with "Test".
TestResult.__test__ = False
TesterAgent.__test__ = False


class MockLLMClient:
    """Mock LLM client for testing that returns queued responses."""

    def __init__(self, responses: list[str] | None = None, settings_instance: Settings | None = None) -> None:
        self.settings = settings_instance or settings
        self._responses: deque[str] = deque(responses or [])
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.call_count += 1
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        if self._responses:
            return self._responses.popleft()
        return '{"summary": "mock", "edits": [], "issues": []}'

    async def read_file(self, path: str | Path) -> str:
        return Path(path).read_text()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    async def edit_file(self, path: str | Path, old: str, new: str) -> None:
        p = Path(path)
        content = p.read_text()
        if old not in content:
            raise ValueError(f"String not found: {old[:80]}")
        content = content.replace(old, new, 1)
        p.write_text(content)

    async def delete_file(self, path: str | Path) -> None:
        Path(path).unlink(missing_ok=True)

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]


@pytest.fixture
def mock_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def mock_client_with_responses():
    def _make(responses: list[str]) -> MockLLMClient:
        return MockLLMClient(responses=responses)

    return _make


@pytest.fixture
def test_settings(tmp_path, monkeypatch) -> Settings:
    """A Settings instance pointing its workspace at a temp directory."""
    s = settings
    monkeypatch.setattr(s, "workspace", tmp_path)
    return s
