from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pytest_asyncio import fixture

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.state import RunState, StateManager, TaskState


@pytest.fixture
def mock_settings() -> Settings:
    """Return a default Settings instance."""
    return Settings()


@pytest.fixture
def sample_plan() -> Plan:
    """Return a Plan with 2 sample tasks."""
    return Plan(
        tasks=[
            TaskModel(id="task-1", description="First task", files=["file1.py"]),
            TaskModel(id="task-2", description="Second task", files=["file2.py"]),
        ],
        rationale="Test plan",
    )


@pytest.fixture
def passing_result() -> TestResult:
    """Return a passing TestResult."""
    return TestResult(passed=True, summary="All tests passed", failures=[])


@pytest.fixture
def failing_result() -> TestResult:
    """Return a failing TestResult."""
    return TestResult(passed=False, summary="Some tests failed", failures=["test_example failed"])


@pytest.fixture
def sample_run_state(sample_plan: Plan) -> RunState:
    """Return a RunState with sample tasks."""
    tasks = [
        TaskState(
            id=task.id,
            description=task.description,
            files=task.files,
            status=task.status,
            result=task.result,
        )
        for task in sample_plan.tasks
    ]
    return RunState.new(goal="Test goal", tasks=tasks)


@pytest.fixture
def state_manager(tmp_path: Path) -> StateManager:
    """Return a StateManager using a temp file."""
    return StateManager(path=tmp_path / ".furrow-state.json")


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: dict[str, str] | None = None, settings: Settings | None = None) -> None:
        self.responses = responses or {}
        self.default_response = '{"tasks": [], "rationale": "mock"}'
        self.call_count = 0
        self.last_prompt: str | None = None
        self.last_system: str | None = None
        self.last_model: str | None = None
        self.settings = settings or Settings()

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system = system
        self.last_model = model

        # Check for exact prompt match first
        if prompt in self.responses:
            return self.responses[prompt]

        # Check for partial matches (key in prompt)
        for key, response in self.responses.items():
            if key in prompt:
                return response

        return self.default_response

    async def read_file(self, path: str | Path) -> str:
        return f"Mock content of {path}"

    async def write_file(self, path: str | Path, content: str) -> None:
        pass

    def list_files(self, directory: str | Path) -> list[str]:
        return ["mock_file.py"]


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Return a MockLLMClient with default settings."""
    return MockLLMClient()


@pytest.fixture
def mock_llm_client_with_responses() -> MockLLMClient:
    """Return a MockLLMClient with predefined responses."""
    return MockLLMClient(
        responses={
            "plan": '{"tasks": [{"id": "1", "description": "Test task", "files": []}], "rationale": "test"}',
            "test": '{"passed": true, "summary": "All good", "failures": []}',
            "worker": "Task completed successfully",
        }
    )