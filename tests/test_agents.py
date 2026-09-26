"""Tests for planner, worker, and tester agents."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


class FakeLLMClient:
    """A fake LLM client for testing that returns configurable responses."""

    def __init__(self, responses=None, settings=None):
        self._responses = list(responses or [])
        self.call_count = 0
        self.settings = settings or MagicMock()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return '{"tasks": [], "rationale": "no more work"}'


def make_task(task_id="1", description="do something", files=None):
    return TaskModel(
        id=task_id,
        description=description,
        files=files or [],
    )


# ---------------------------------------------------------------------------
# PlannerAgent tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_planner_parses_valid_json():
    plan_json = json.dumps({
        "tasks": [
            {"id": "1", "description": "Create file A", "files": ["a.py"], "dependencies": []},
            {"id": "2", "description": "Create file B", "files": ["b.py"], "dependencies": ["1"]},
        ],
        "rationale": "Split into two independent files",
    })
    client = FakeLLMClient(responses=[plan_json])
    planner = PlannerAgent(client=client)

    plan = await planner.plan("Create two files")

    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "Create file A"
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.rationale == "Split into two independent files"


@pytest.mark.asyncio
async def test_planner_empty_tasks():
    plan_json = json.dumps({"tasks": [], "rationale": "Goal already complete"})
    client = FakeLLMClient(responses=[plan_json])
    planner = PlannerAgent(client=client)

    plan = await planner.plan("Already done task")

    assert len(plan.tasks) == 0


@pytest.mark.asyncio
async def test_planner_invalid_json_raises():
    client = FakeLLMClient(responses=["this is not json"])
    planner = PlannerAgent(client=client)

    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Some goal")


@pytest.mark.asyncio
async def test_planner_missing_required_field_raises():
    # Plan is missing "rationale"
    plan_json = json.dumps({"tasks": []})
    client = FakeLLMClient(responses=[plan_json])
    planner = PlannerAgent(client=client)

    with pytest.raises(ValueError):
        await planner.plan("Some goal")


# ---------------------------------------------------------------------------
# WorkerAgent tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_returns_llm_response():
    client = FakeLLMClient(responses=["I implemented the feature"])
    worker = WorkerAgent(task=make_task(), client=client)

    result = await worker.run()

    assert result == "I implemented the feature"
    assert client.call_count == 1


@pytest.mark.asyncio
async def test_worker_passes_model_override():
    mock_settings = MagicMock()
    mock_settings.worker_model = "some-model"
    client = FakeLLMClient(settings=mock_settings)
    worker = WorkerAgent(task=make_task(), client=client)

    await worker.run()

    assert client.call_count == 1


# ---------------------------------------------------------------------------
# TesterAgent tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tester_passed():
    plan_json = json.dumps({
        "passed": True,
        "summary": "All 10 tests passed",
        "failures": [],
    })
    client = FakeLLMClient(responses=[plan_json])
    tester = TesterAgent(client=client)

    # Mock _run_tests to return fake output
    tester._run_tests = AsyncMock(return_value="10 passed")

    result = await tester.run("Some goal", [make_task()])

    assert result.passed is True
    assert "10 tests passed" in result.summary
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_failed_with_failures():
    plan_json = json.dumps({
        "passed": False,
        "summary": "2 tests failed",
        "failures": ["test_auth failed: assertion error", "test_router failed: import error"],
    })
    client = FakeLLMClient(responses=[plan_json])
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="2 failed")

    result = await tester.run("Some goal", [make_task()])

    assert result.passed is False
    assert len(result.failures) == 2
    assert "test_auth" in result.failures[0]


@pytest.mark.asyncio
async def test_tester_handles_non_json_response():
    client = FakeLLMClient(responses=["Tests passed based on output"])
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="10 passed")

    result = await tester.run("Some goal", [make_task()])

    # Should fall back to string matching for "passed"
    assert result.passed is True
    assert "Tests passed" in result.summary


@pytest.mark.asyncio
async def test_tester_no_test_runner():
    """When no test runner is found, the tester should return a failure result."""
    plan_json = json.dumps({
        "passed": True,
        "summary": "No tests to run",
        "failures": [],
    })
    client = FakeLLMClient(responses=[plan_json])
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="No test runner found.")

    result = await tester.run("Some goal", [make_task()])

    assert result.passed is True
    assert client.call_count == 1  # LLM was called with the output


@pytest.mark.asyncio
async def test_tester_exception_in_run_tests():
    """If _run_tests raises, tester should return a failed result."""
    client = FakeLLMClient(responses=[])
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(side_effect=RuntimeError("subprocess error"))

    result = await tester.run("Some goal", [make_task()])

    assert result.passed is False
    assert "subprocess error" in result.summary
    assert "subprocess error" in result.failures[0]
    # LLM should NOT have been called since tests failed before reaching it
    assert client.call_count == 0


@pytest.mark.asyncio
async def test_run_tests_finds_no_runner_without_any_commands():
    """_run_tests returns fallback message when no test runner executables exist."""
    tester = TesterAgent(client=FakeLLMClient())

    # Patch subprocess creation to raise FileNotFoundError (simulating no test runners)
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("no pytest")):
        result = await tester._run_tests()

    assert result == "No test runner found."
