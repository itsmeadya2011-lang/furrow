"""Tests for the Orchestrator's core loop logic."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from tests.conftest import MockLLMClient


# ---------------------------------------------------------------------------
# _is_done tests
# ---------------------------------------------------------------------------


def test_is_done_no_tasks_no_test_result():
    """No tasks and no test result — goal is done (nothing to do)."""
    orch = Orchestrator(goal="test", client=MockLLMClient())
    assert orch._is_done() is True


def test_is_done_no_tasks_tests_passed():
    """No tasks but tests passed — done."""
    orch = Orchestrator(goal="test", client=MockLLMClient())
    orch._last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


def test_is_done_no_tasks_tests_failed():
    """No tasks but tests failed — not done (will retry)."""
    orch = Orchestrator(goal="test", client=MockLLMClient())
    orch._last_test_result = TestResult(passed=False, summary="fail", failures=["err"])
    assert orch._is_done() is False


def test_is_done_all_tasks_completed_tests_passed():
    """All tasks completed and tests passed — done."""
    orch = Orchestrator(goal="test", client=MockLLMClient())
    orch._tasks = [TaskModel(id="1", description="task", status="completed")]
    orch._last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


def test_is_done_task_failed():
    """Any task failed — not done."""
    orch = Orchestrator(goal="test", client=MockLLMClient())
    orch._tasks = [
        TaskModel(id="1", description="ok", status="completed"),
        TaskModel(id="2", description="bad", status="failed", result="error"),
    ]
    orch._last_test_result = TestResult(passed=False, summary="fail", failures=["err"])
    assert orch._is_done() is False


def test_is_done_all_tasks_completed_tests_failed():
    """Tasks completed but tests failed — not done."""
    orch = Orchestrator(goal="test", client=MockLLMClient())
    orch._tasks = [TaskModel(id="1", description="task", status="completed")]
    orch._last_test_result = TestResult(passed=False, summary="fail", failures=["err"])
    assert orch._is_done() is False


def test_is_done_task_pending():
    """Task still pending — not done."""
    orch = Orchestrator(goal="test", client=MockLLMClient())
    orch._tasks = [TaskModel(id="1", description="task", status="pending")]
    orch._last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is False


# ---------------------------------------------------------------------------
# _get_tasks tests
# ---------------------------------------------------------------------------


def test_get_tasks_returns_stored_tasks():
    """_get_tasks should return the tasks stored from the last cycle."""
    orch = Orchestrator(goal="test", client=MockLLMClient())
    tasks = [TaskModel(id="1", description="task1"), TaskModel(id="2", description="task2")]
    orch._tasks = tasks
    assert orch._get_tasks() == tasks


def test_get_tasks_empty_by_default():
    """_get_tasks should return empty list before any cycle runs."""
    orch = Orchestrator(goal="test", client=MockLLMClient())
    assert orch._get_tasks() == []


# ---------------------------------------------------------------------------
# max_cycles tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_cycles_enforced():
    """Orchestrator should stop after max_cycles even if _is_done returns False."""
    settings = Settings(max_cycles=2)
    mock = MockLLMClient(settings=settings)
    orch = Orchestrator(goal="test", client=mock)

    orch._cycle = AsyncMock()
    orch._is_done = lambda: False

    await orch.run()
    assert orch.cycles == 2
    assert orch._cycle.call_count == 2


@pytest.mark.asyncio
async def test_max_cycles_zero_is_infinite():
    """max_cycles=0 means infinite — but _is_done stops it."""
    settings = Settings(max_cycles=0)
    mock = MockLLMClient(settings=settings)
    orch = Orchestrator(goal="test", client=mock)

    call_count = 0

    async def mock_cycle():
        nonlocal call_count
        call_count += 1
        if call_count >= 1:
            orch._tasks = [TaskModel(id="1", description="task", status="completed")]
            orch._last_test_result = TestResult(passed=True, summary="ok", failures=[])

    orch._cycle = mock_cycle
    await orch.run()
    assert call_count == 1


# ---------------------------------------------------------------------------
# Output callback tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_callback_receives_messages():
    """The on_output callback should receive plain text versions of emitted messages."""
    received = []

    async def on_output(text: str) -> None:
        received.append(text)

    orch = Orchestrator(goal="test", client=MockLLMClient(), on_output=on_output)
    orch._cycle = AsyncMock()
    orch._is_done = lambda: True

    await orch.run()
    assert len(received) > 0
    assert any("Goal: test" in msg for msg in received)
    assert any("Cycle 1" in msg for msg in received)


@pytest.mark.asyncio
async def test_no_callback_works():
    """Orchestrator should work fine without an output callback."""
    orch = Orchestrator(goal="test", client=MockLLMClient())
    orch._cycle = AsyncMock()
    orch._is_done = lambda: True

    await orch.run()
    assert orch.cycles == 1


# ---------------------------------------------------------------------------
# _cycle integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cycle_stores_tasks_and_test_results():
    """_cycle should store tasks in _tasks and test result in _last_test_result."""
    plan_json = json.dumps({
        "tasks": [
            {"id": "t1", "description": "Task 1", "files": [], "dependencies": []},
            {"id": "t2", "description": "Task 2", "files": [], "dependencies": []},
        ],
        "rationale": "Test plan",
    })
    test_json = json.dumps({
        "passed": True,
        "summary": "All tests passed",
        "failures": [],
    })

    mock = MockLLMClient(responses=[plan_json, "Task 1 done", "Task 2 done", test_json])
    orch = Orchestrator(goal="Build feature", client=mock)

    with patch("furrow.agents.tester.TesterAgent._run_tests", new_callable=AsyncMock, return_value="5 passed"):
        await orch._cycle()

    assert len(orch._tasks) == 2
    assert orch._tasks[0].status == "completed"
    assert orch._tasks[1].status == "completed"
    assert orch._tasks[0].result == "Task 1 done"
    assert orch._tasks[1].result == "Task 2 done"
    assert orch._last_test_result is not None
    assert orch._last_test_result.passed is True


@pytest.mark.asyncio
async def test_cycle_with_test_failure_rewrites_goal():
    """_cycle should rewrite goal on test failure."""
    plan_json = json.dumps({
        "tasks": [{"id": "t1", "description": "Fix the bug", "files": [], "dependencies": []}],
        "rationale": "Fix plan",
    })
    test_json = json.dumps({
        "passed": False,
        "summary": "2 failures",
        "failures": ["test_foo.py::test_foo failed", "test_bar.py::test_bar failed"],
    })

    mock = MockLLMClient(responses=[plan_json, "Fixed bug", test_json])
    orch = Orchestrator(goal="Build feature", client=mock)

    with patch("furrow.agents.tester.TesterAgent._run_tests", new_callable=AsyncMock, return_value="2 failed"):
        await orch._cycle()

    assert orch._last_test_result.passed is False
    assert "Fix failing tests:" in orch.goal
    assert "test_foo.py::test_foo failed" in orch.goal


@pytest.mark.asyncio
async def test_cycle_with_worker_exception():
    """_cycle should mark tasks as failed if the worker raises an exception."""
    plan_json = json.dumps({
        "tasks": [{"id": "t1", "description": "Task 1", "files": [], "dependencies": []}],
        "rationale": "Test plan",
    })
    mock = MockLLMClient(responses=[plan_json])
    orch = Orchestrator(goal="test", client=mock)

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
        async def _fail_run():
            raise RuntimeError("Worker crashed")

        MockWorker.return_value.run = _fail_run

        with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            MockTester.return_value.run = AsyncMock(
                return_value=TestResult(passed=True, summary="ok", failures=[])
            )
            await orch._cycle()

    assert orch._tasks[0].status == "failed"
    assert "Worker crashed" in (orch._tasks[0].result or "")
