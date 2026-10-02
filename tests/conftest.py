"""Shared test fixtures and mocks for Furrow tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from furrow.config import Provider, Settings


class MockLLMClient:
    """A mock LLM client that returns pre-configured responses without API calls."""

    def __init__(
        self,
        responses: list[str] | str | dict[str, str] | None = None,
        settings: Settings | None = None,
        existing_files: dict[str, str] | None = None,
    ) -> None:
        self._responses = responses or []
        self._response_idx = 0
        self._response_map = {}
        if isinstance(responses, dict):
            self._response_map = responses
        self.settings = settings or Settings(
            provider=Provider.ANTHROPIC,
            model="test-model",
            planner_model="test-planner",
            worker_model="test-worker",
            tester_model="test-tester",
            max_cycles=0,
            workspace=Path("/tmp/furrow_test_workspace"),
        )
        self.written_files: dict[str, str] = {}
        self.existing_files: dict[str, str] = existing_files or {}
        self.complete_call_count = 0
        self.complete_calls: list[tuple[str, str]] = []

    def _normalize_path(self, path: str | Path) -> str:
        """Normalize to a workspace-relative path so keys are consistent."""
        p = Path(path)
        workspace = Path(self.settings.workspace)
        try:
            return str(p.relative_to(workspace))
        except ValueError:
            return str(p)

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None, timeout: float | None = None
    ) -> str:
        self.complete_call_count += 1
        self.complete_calls.append((prompt, system))
        if self._response_map:
            # Try to match by prompt keyword
            for key, value in self._response_map.items():
                if key in prompt:
                    return value
            return json.dumps({"changes": [], "summary": "No match"})
        if isinstance(self._responses, list) and self._responses:
            if self._response_idx < len(self._responses):
                resp = self._responses[self._response_idx]
                self._response_idx += 1
                return resp
            return self._responses[-1]  # repeat last
        return str(self._responses)

    async def read_file(self, path: str | Path) -> str:
        key = self._normalize_path(path)
        if key in self.existing_files:
            return self.existing_files[key]
        raise FileNotFoundError(f"Mock: file not found: {path}")

    async def write_file(self, path: str | Path, content: str) -> None:
        key = self._normalize_path(path)
        self.written_files[key] = content

    def list_files(self, directory: str | Path) -> list[str]:
        return list(self.existing_files.keys())


@pytest.fixture
def mock_client() -> MockLLMClient:
    return MockLLMClient()
