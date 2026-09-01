from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value="mocked response")
    client.settings = Settings(
        provider=Provider.ANTHROPIC,
        model="test-model",
        planner_model="test-planner",
        worker_model="test-worker",
        tester_model="test-tester",
        anthropic_api_key="test-key",
    )
    return client


@pytest.fixture
def test_settings():
    return Settings(
        provider=Provider.ANTHROPIC,
        model="test-model",
        planner_model="test-planner",
        worker_model="test-worker",
        tester_model="test-tester",
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
        ollama_base_url="http://localhost:11434",
        max_parallel_tasks=3,
        max_cycles=5,
        workspace=Path("/tmp/test-workspace"),
        log_level="DEBUG",
    )


@pytest.fixture
def sample_task():
    return TaskModel(
        id="task-1",
        description="Implement authentication",
        files=["src/auth.py", "tests/test_auth.py"],
        dependencies=[],
        status="pending",
    )


@pytest.fixture
def sample_task_completed():
    return TaskModel(
        id="task-2",
        description="Add logging",
        files=["src/logging.py"],
        dependencies=[],
        status="completed",
        result="Added logging module",
    )


@pytest.fixture
def sample_task_failed():
    return TaskModel(
        id="task-3",
        description="Fix bug",
        files=["src/bug.py"],
        dependencies=[],
        status="failed",
        result="Error: something went wrong",
    )


@pytest.fixture
def sample_plan():
    return Plan(
        tasks=[
            TaskModel(id="1", description="Task one", files=["a.py"]),
            TaskModel(id="2", description="Task two", files=["b.py"]),
        ],
        rationale="Test plan rationale",
    )


@pytest.fixture
def sample_test_result_passed():
    return TestResult(passed=True, summary="All tests passed", failures=[])


@pytest.fixture
def sample_test_result_failed():
    return TestResult(
        passed=False,
        summary="Some tests failed",
        failures=["test_auth failed", "test_db failed"],
    )
