"""Tests for the Furrow agents (Planner, Worker, Tester)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from tests.conftest import MockLLMClient


# ---------------------------------------------------------------------------
# PlannerAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_returns_valid_plan():
    """PlannerAgent should parse valid JSON into a Plan."""
    plan_json = json.dumps({
        "tasks": [
            {
                "id": "1",
                "description": "Create user model",
                "files": ["models/user.py"],
                "dependencies": [],
            },
            {
                "id": "2",
                "description": "Add auth routes",
                "files": ["routes/auth.py", "tests/test_auth.py"],
                "dependencies": ["1"],
            },
        ],
        "rationale": "Start with the model then add auth",
    })
    mock = MockLLMClient(responses=[plan_json])
    planner = PlannerAgent(client=mock)

    plan = await planner.plan("Build user authentication")

    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "Create user model"
    assert plan.tasks[0].files == ["models/user.py"]
    assert plan.tasks[1].dependencies == ["1"]
    assert "Start with the model" in plan.rationale


@pytest.mark.asyncio
async def test_planner_empty_plan():
    """PlannerAgent should handle a plan with no tasks."""
    plan_json = json.dumps({"tasks": [], "rationale": "Goal already complete"})
    mock = MockLLMClient(responses=[plan_json])
    planner = PlannerAgent(client=mock)

    plan = await planner.plan("Check if anything needs doing")

    assert len(plan.tasks) == 0
    assert "Goal already complete" in plan.rationale


@pytest.mark.asyncio
async def test_planner_invalid_json_raises():
    """PlannerAgent should raise ValueError on invalid JSON."""
    mock = MockLLMClient(responses=["This is not JSON"])
    planner = PlannerAgent(client=mock)

    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Some goal")


@pytest.mark.asyncio
async def test_planner_passes_correct_model():
    """PlannerAgent should use the planner_model from settings."""
    plan_json = json.dumps({"tasks": [], "rationale": "ok"})
    mock = MockLLMClient(responses=[plan_json])
    planner = PlannerAgent(client=mock)

    await planner.plan("goal")

    assert mock.call_log[0]["model"] == mock.settings.planner_model


# ---------------------------------------------------------------------------
# WorkerAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_returns_summary():
    """WorkerAgent should return the LLM response."""
    mock = MockLLMClient(responses=["Created models/user.py with fields id, name, email"])
    task = TaskModel(id="1", description="Create user model", files=["models/user.py"])
    worker = WorkerAgent(task=task, client=mock)

    result = await worker.run()

    assert "models/user.py" in result
    assert mock.call_log[0]["model"] == mock.settings.worker_model


@pytest.mark.asyncio
async def test_worker_includes_files_in_prompt():
    """WorkerAgent should include task files in the prompt context."""
    mock = MockLLMClient(responses=["done"])
    task = TaskModel(
        id="1",
        description="Update routes",
        files=["routes/auth.py", "routes/users.py"],
    )
    worker = WorkerAgent(task=task, client=mock)

    await worker.run()

    prompt = mock.call_log[0]["prompt"]
    assert "routes/auth.py" in prompt
    assert "routes/users.py" in prompt


# ---------------------------------------------------------------------------
# TesterAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tester_passes_with_no_failures():
    """TesterAgent should return a passing TestResult when tests pass."""
    test_json = json.dumps({
        "passed": True,
        "summary": "All 5 tests passed",
        "failures": [],
    })
    mock = MockLLMClient(responses=[test_json])
    tester = TesterAgent(client=mock)

    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="5 passed"):
        result = await tester.run("Build auth", [])

    assert result.passed is True
    assert "5 tests" in result.summary
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_detects_failures():
    """TesterAgent should return a failing TestResult when tests fail."""
    test_json = json.dumps({
        "passed": False,
        "summary": "2 tests failed",
        "failures": ["test_auth.py::test_login failed", "test_auth.py::test_logout failed"],
    })
    mock = MockLLMClient(responses=[test_json])
    tester = TesterAgent(client=mock)

    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="2 failed"):
        result = await tester.run("Build auth", [])

    assert result.passed is False
    assert len(result.failures) == 2


@pytest.mark.asyncio
async def test_tester_handles_json_parse_error():
    """TesterAgent should fall back to string matching if JSON parse fails."""
    mock = MockLLMClient(responses=["All tests passed successfully"])
    tester = TesterAgent(client=mock)

    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="1 passed"):
        result = await tester.run("Build auth", [])

    assert result.passed is True  # "passed" found in response


@pytest.mark.asyncio
async def test_tester_handles_test_runner_exception():
    """TesterAgent should return a failing result if test execution itself fails."""
    mock = MockLLMClient(responses=["anything"])
    tester = TesterAgent(client=mock)

    async def _raise():
        raise RuntimeError("pytest not found")

    with patch.object(tester, "_run_tests", new_callable=AsyncMock, side_effect=_raise):
        result = await tester.run("Build auth", [])

    assert result.passed is False
    assert "pytest not found" in result.summary
