"""Tests for the Furrow agents: Planner, Worker, and Tester."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_parses_valid_json():
    """PlannerAgent.plan() should parse a valid JSON response into a Plan."""
    plan_data = {
        "tasks": [
            {
                "id": "1",
                "description": "Implement feature X",
                "files": ["src/feature.py", "tests/test_feature.py"],
                "dependencies": [],
            }
        ],
        "rationale": "Add feature X for the goal",
    }
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=json.dumps(plan_data))
    mock_client.settings = MagicMock()

    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("Add feature X")

    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "Implement feature X"
    assert plan.tasks[0].files == ["src/feature.py", "tests/test_feature.py"]
    assert plan.rationale == "Add feature X for the goal"


@pytest.mark.asyncio
async def test_planner_raises_on_invalid_json():
    """PlannerAgent.plan() should raise ValueError on invalid JSON response."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="This is not JSON")
    mock_client.settings = MagicMock()

    planner = PlannerAgent(client=mock_client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Some goal")


@pytest.mark.asyncio
async def test_planner_raises_on_missing_required_field():
    """PlannerAgent.plan() should raise when required fields are missing."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value='{"rationale": "missing tasks"}')
    mock_client.settings = MagicMock()

    planner = PlannerAgent(client=mock_client)
    with pytest.raises(ValueError):
        await planner.plan("Some goal")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_returns_llm_response():
    """WorkerAgent.run() should return the LLM response string."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="Implemented the feature")
    mock_client.settings = MagicMock(worker_model="test-model")

    task = TaskModel(id="1", description="Do something", files=["src/main.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert result == "Implemented the feature"
    mock_client.complete.assert_awaited_once()
    call_kwargs = mock_client.complete.call_args
    prompt = call_kwargs.args[0]
    assert "Do something" in prompt
    assert "src/main.py" in prompt


@pytest.mark.asyncio
async def test_worker_handles_empty_files():
    """WorkerAgent.run() should handle tasks with no files specified."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="Done")
    mock_client.settings = MagicMock(worker_model="test-model")

    task = TaskModel(id="1", description="Refactor module")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert result == "Done"
    prompt = mock_client.complete.call_args.args[0]
    assert "any" in prompt  # "Files to touch: any" when empty


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tester_returns_passed_result():
    """TesterAgent.run() should return a passed TestResult when tests pass."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(
        return_value=json.dumps({"passed": True, "summary": "All tests passed", "failures": []})
    )
    mock_client.settings = MagicMock()

    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new=AsyncMock(return_value="=== 3 passed")):
        result = await tester.run("Make tests pass", [])

    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_returns_failed_result():
    """TesterAgent.run() should return a failed TestResult with failures."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(
        return_value=json.dumps({
            "passed": False,
            "summary": "2 failures",
            "failures": ["test_foo failed", "test_bar failed"],
        })
    )
    mock_client.settings = MagicMock()

    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new=AsyncMock(return_value="=== 2 failed")):
        result = await tester.run("Fix tests", [])

    assert result.passed is False
    assert len(result.failures) == 2
    assert "test_foo" in result.failures[0]


@pytest.mark.asyncio
async def test_tester_fallback_preserves_failures():
    """TesterAgent.run() should include raw response in failures when JSON parse fails and tests failed."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="Tests look broken but no JSON")
    mock_client.settings = MagicMock()

    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new=AsyncMock(return_value="=== failures")):
        result = await tester.run("Fix tests", [])

    assert result.passed is False
    assert len(result.failures) == 1
    assert "Tests look broken but no JSON" in result.failures[0]


@pytest.mark.asyncio
async def test_tester_fallback_passed_when_no_passed_keyword():
    """TesterAgent.run() should assume passed=False when JSON fails and 'passed' not in response."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="some garbage output with no relevant keywords")
    mock_client.settings = MagicMock()

    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new=AsyncMock(return_value="some output")):
        result = await tester.run("Goal", [])

    assert result.passed is False
    assert len(result.failures) == 1
    assert result.failures[0] == "some garbage output with no relevant keywords"


@pytest.mark.asyncio
async def test_tester_handles_test_runner_exception():
    """TesterAgent.run() should return a failed TestResult when the test runner itself fails."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({"passed": True, "summary": "ok", "failures": []}))
    mock_client.settings = MagicMock()

    tester = TesterAgent(client=mock_client)
    with patch.object(
        tester, "_run_tests", new=AsyncMock(side_effect=RuntimeError("No test runner found"))
    ):
        result = await tester.run("Goal", [])

    assert result.passed is False
    assert "No test runner found" in result.summary


@pytest.mark.asyncio
async def test_tester_no_runner_returns_fallback():
    """TesterAgent._run_tests() should return a message when no test runner is found."""
    mock_client = MagicMock()
    mock_client.settings = MagicMock()

    tester = TesterAgent(client=mock_client)

    with patch("furrow.agents.tester.asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        result = await tester._run_tests()

    assert result == "No test runner found."
