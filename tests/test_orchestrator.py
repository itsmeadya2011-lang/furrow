"""Tests for the Orchestrator loop and task tracking."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _plan_response(tasks: list[dict]) -> str:
    """Serialize a plan as the LLM would return it."""
    return json.dumps({"tasks": tasks, "rationale": "test plan"})


def _test_result_response(passed: bool, summary: str, failures: list[str] | None = None) -> str:
    return json.dumps({"passed": passed, "summary": summary, "failures": failures or []})


@pytest.fixture(autouse=True)
def _mock_test_runner():
    """Prevent real test subprocess execution during orchestrator tests."""
    with patch.object(
        TesterAgent, "_run_tests", new_callable=AsyncMock
    ) as mock:
        mock.return_value = "All tests passed\n"
        yield mock


@pytest.mark.asyncio
async def test_orchestrator_success_single_cycle(mock_llm_client: MagicMock):
    """Goal completes in one cycle when tests pass."""
    mock_llm_client.complete.side_effect = [
        # Planner response
        _plan_response([{"id": "1", "description": "Add a README section", "files": ["README.md"]}]),
        # Worker response
        "Added a README section.",
        # Tester response (pass)
        _test_result_response(True, "All tests passed"),
    ]

    orch = Orchestrator(goal="Improve README", client=mock_llm_client)
    await orch.run()

    assert orch.cycles == 1
    assert len(orch.all_tasks) == 1
    assert orch.all_tasks[0].status == "completed"
    assert orch.last_test_result is not None
    assert orch.last_test_result.passed is True


@pytest.mark.asyncio
async def test_orchestrator_continues_when_tests_fail(mock_llm_client: MagicMock):
    """Loop continues to a second cycle when tests fail."""
    mock_llm_client.complete.side_effect = [
        # Cycle 1: planner
        _plan_response([{"id": "1", "description": "Fix bug", "files": ["src/bug.py"]}]),
        # Cycle 1: worker
        "Fixed bug.",
        # Cycle 1: tester (fail)
        _test_result_response(False, "1 test failed", ["test_bug.py::test_1 failed"]),
        # Cycle 2: planner
        _plan_response([{"id": "2", "description": "Re-fix bug", "files": ["src/bug.py"]}]),
        # Cycle 2: worker
        "Re-fixed bug.",
        # Cycle 2: tester (pass)
        _test_result_response(True, "All tests passed"),
    ]

    orch = Orchestrator(goal="Fix the bug", client=mock_llm_client)
    await orch.run()

    assert orch.cycles == 2
    assert len(orch.all_tasks) == 2  # tasks accumulated across cycles


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles(mock_llm_client: MagicMock):
    """Loop stops when max_cycles is reached, even if tests fail."""
    mock_llm_client.settings.max_cycles = 2

    mock_llm_client.complete.side_effect = [
        # Cycle 1
        _plan_response([{"id": "1", "description": "Task", "files": []}]),
        "done",
        _test_result_response(False, "failed", ["error"]),
        # Cycle 2
        _plan_response([{"id": "2", "description": "Task 2", "files": []}]),
        "done",
        _test_result_response(False, "failed", ["error"]),
    ]

    orch = Orchestrator(goal="Keep failing", client=mock_llm_client)
    await orch.run()

    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_orchestrator_worker_exception_handled(mock_llm_client: MagicMock):
    """A worker exception is caught by gather(return_exceptions=True); task marked failed."""
    mock_llm_client.complete.side_effect = [
        # Planner
        _plan_response([{"id": "1", "description": "Task", "files": []}]),
        # Worker — raises an exception (caught by asyncio.gather with return_exceptions=True)
        RuntimeError("Worker boom"),
        # Tester (still runs regardless of worker outcome)
        _test_result_response(True, "All tests passed"),
    ]

    orch = Orchestrator(goal="Test", client=mock_llm_client)
    await orch.run()

    assert orch.all_tasks[0].status == "failed"
    assert "Worker boom" in orch.all_tasks[0].result
    assert orch.last_test_result.passed is True


@pytest.mark.asyncio
async def test_orchestrator_no_tasks_planned(mock_llm_client: MagicMock):
    """If planner returns no tasks, cycle is skipped, last_test_result stays None."""
    mock_llm_client.complete.return_value = _plan_response([])
    mock_llm_client.settings.max_cycles = 1

    orch = Orchestrator(goal="Already done", client=mock_llm_client)
    await orch.run()

    assert orch.cycles == 1
    assert orch.last_test_result is None


@pytest.mark.asyncio
async def test_orchestrator_goal_rewritten_on_failure(mock_llm_client: MagicMock):
    """Goal string is updated to include failure details after a failed test cycle."""
    mock_llm_client.complete.side_effect = [
        _plan_response([{"id": "1", "description": "Task", "files": []}]),
        "done",
        _test_result_response(False, "broken", ["test_x failed: assertion error"]),
        # Second cycle (to continue loop then stop)
        _plan_response([{"id": "2", "description": "Fix", "files": []}]),
        "done",
        _test_result_response(True, "All passed"),
    ]

    orch = Orchestrator(goal="Original goal", client=mock_llm_client)
    await orch.run()

    # Goal should have been rewritten to include failure info
    assert "Fix failing tests" in orch.goal
    assert "test_x failed" in orch.goal


@pytest.mark.asyncio
async def test_orchestrator_multiple_tasks_parallel(mock_llm_client: MagicMock):
    """Multiple tasks in a single cycle all get executed."""
    mock_llm_client.complete.side_effect = [
        # Planner — 3 tasks
        _plan_response([
            {"id": "1", "description": "Task A", "files": ["a.py"]},
            {"id": "2", "description": "Task B", "files": ["b.py"]},
            {"id": "3", "description": "Task C", "files": ["c.py"]},
        ]),
        # Worker A
        "A done",
        # Worker B
        "B done",
        # Worker C
        "C done",
        # Tester
        _test_result_response(True, "All passed"),
    ]

    orch = Orchestrator(goal="Do everything", client=mock_llm_client)
    await orch.run()

    assert orch.cycles == 1
    assert len(orch.all_tasks) == 3
    assert all(t.status == "completed" for t in orch.all_tasks)


@pytest.mark.asyncio
async def test_orchestrator_worker_model_usage(mock_llm_client: MagicMock):
    """Worker uses worker_model; planner uses planner_model (verified via call args)."""
    mock_llm_client.complete.side_effect = [
        _plan_response([{"id": "1", "description": "Task", "files": []}]),
        "done",
        _test_result_response(True, "ok"),
    ]

    orch = Orchestrator(goal="Test model dispatch", client=mock_llm_client)
    await orch.run()

    calls = mock_llm_client.complete.call_args_list
    # First call is planner, using planner_model
    assert calls[0].kwargs["model"] == mock_llm_client.settings.planner_model
    # Second call is worker, using worker_model
    assert calls[1].kwargs["model"] == mock_llm_client.settings.worker_model
    # Third call is tester, using tester_model
    assert calls[2].kwargs["model"] == mock_llm_client.settings.tester_model


def test_orchestrator_completed_and_failed_properties(mock_llm_client: MagicMock):
    """The completed_tasks and failed_tasks properties filter by status."""
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="A", status="completed"),
            TaskModel(id="2", description="B", status="completed"),
            TaskModel(id="3", description="C", status="failed", result="error"),
        ],
        rationale="test",
    )

    orch = Orchestrator(goal="test", client=mock_llm_client)
    orch.all_tasks = plan.tasks

    assert len(orch.completed_tasks) == 2
    assert len(orch.failed_tasks) == 1


def test_orchestrator_is_done_no_test_result(mock_llm_client: MagicMock):
    """_is_done returns False before any test run."""
    orch = Orchestrator(goal="test", client=mock_llm_client)
    assert orch._is_done() is False


def test_orchestrator_is_done_test_passed(mock_llm_client: MagicMock):
    """_is_done returns True when tests pass."""
    orch = Orchestrator(goal="test", client=mock_llm_client)
    orch.last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is True


def test_orchestrator_is_done_test_failed(mock_llm_client: MagicMock):
    """_is_done returns False when tests fail."""
    orch = Orchestrator(goal="test", client=mock_llm_client)
    orch.last_test_result = TestResult(passed=False, summary="bad", failures=["fail"])
    assert orch._is_done() is False
