from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from furrow.config import Provider, Settings, TaskModel
from furrow.llm import LLMClient


class MockLLMClient(LLMClient):
    """Test double that subclasses LLMClient and returns canned completions."""

    def __init__(
        self,
        settings: Settings | None = None,
        responses: list[str] | None = None,
    ) -> None:
        self.responses: list[str] = list(responses or [])
        self.complete_calls: list[dict[str, Any]] = []
        super().__init__(settings=settings or _default_settings())

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
    ) -> str:
        self.complete_calls.append({"prompt": prompt, "system": system, "model": model})
        if self.responses:
            return self.responses.pop(0)
        return '{"tasks": [], "rationale": "default mock response"}'


def _default_settings() -> Settings:
    return Settings(
        provider=Provider.ANTHROPIC,
        model="mock-model",
        planner_model="mock-model",
        worker_model="mock-model",
        tester_model="mock-model",
        max_cycles=0,
        max_parallel_tasks=2,
        workspace=Path.cwd(),
    )


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """A temporary directory to use as the LLMClient workspace."""
    return tmp_path


@pytest.fixture
def settings(tmp_workspace: Path) -> Settings:
    """Settings pointing at a temporary workspace; no real API keys required."""
    return Settings(
        provider=Provider.ANTHROPIC,
        model="mock-model",
        planner_model="mock-model",
        worker_model="mock-model",
        tester_model="mock-model",
        max_cycles=0,
        max_parallel_tasks=2,
        workspace=tmp_workspace,
    )


@pytest.fixture
def mock_client(settings: Settings) -> MockLLMClient:
    """A MockLLMClient bound to a temporary workspace."""
    return MockLLMClient(settings=settings)


@pytest.fixture
def task_model() -> TaskModel:
    return TaskModel(id="1", description="implement feature", files=[])
