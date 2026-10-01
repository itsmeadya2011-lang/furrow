"""Tests for the Orchestrator core logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _make_client(tasks: list[TaskModel], test_result: TestResult):
    """Create a mock LLMClient with stubbed planner/worker/tester."""
    client = MagicMock()
    client.settings.max_cycles = 0  # unlimited by default
    client.settings.max_parallel_tasks = 5
    client.settings.planner_model = "planner"
    client.settings.worker_model = "worker"
    client.settings.tester_model = "tester"
    client.settings.provider = MagicMock()

    plan = Plan(tasks=tasks, rationale="test plan")

    planner = AsyncMock()
    planner.plan = AsyncMock(return_value=plan)

    worker_run = AsyncMock(return_value="Summary of changes")

    tester = AsyncMock()
    tester.run = AsyncMock(return_value=test_result)

    client.planner = planner
    client.tasks = tasks  # not used directly
    return client, planner, worker_run, tester


@pytest.mark.asyncio
async def test_run_completes_when_tests_pass():
    tasks = [TaskModel(id="1", description="Create a file")]
    test_result = TestResult(passed=True, summary="All good")
    client, planner, worker_run, tester = _make_client(tasks, test_result)

    # Patch the orchestrator internals to use our mocks.
    orch = Orchestrator(goal="Create a file", client=client)
    orch.planner = planner
    with patch.object(__import__("furrow.core.orchestrator", fromlist=["WorkerAgent"]), "WorkerAgent") as MockWA:
        MockWA.return_value.run = worker_run
        with patch.object(__import__("furrow.core.orchestrator", fromlist=["TesterAgent"]), "TesterAgent") as MockTA:
            MockTA.return_value.run = tester.run
            await orch.run()

    assert orch.cycles == 1
    assert orch._done is True
    planner.plan.assert_called_once()
    worker_run.assert_awaited_once()
    tester.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_max_cycles_limit():
    tasks = [TaskModel(id="1", description="Create a file")]
    test_result = TestResult(passed=False, summary="fail", failures=["broken"])
    client, planner, worker_run, tester = _make_client(tasks, test_result)
    client.settings.max_cycles = 2

    orch = Orchestrator(goal="test", client=client)
    orch.planner = planner
    with patch("furrow.core.orchestrator.WorkerAgent") as MockWA:
        MockWA.return_value.run = worker_run
        with patch("furrow.core.orchestrator.TesterAgent") as MockTA:
            MockTA.return_value.run = tester.run
            await orch.run()

    # With max_cycles=2, should stop after 2 cycles.
    assert orch.cycles == 2
    planner.plan.assert_called()  # called at least once


def test_is_done_no_tasks():
    client = MagicMock()
    orch = Orchestrator(goal="test", client=client)
    orch.tasks = []
    assert orch._is_done() is True


def test_is_done_all_completed_no_tests_passed():
    """All tasks completed but tests haven't passed → keep cycling (not done)."""
    client = MagicMock()
    orch = Orchestrator(goal="test", client=client)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    orch._done = False
    # _done is only set True when tests pass; all-completed without tests
    # passing means we should keep going to fix issues.
    assert orch._is_done() is False


def test_is_done_has_failed():
    client = MagicMock()
    orch = Orchestrator(goal="test", client=client)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    orch._done = False
    assert orch._is_done() is False


def test_is_done_done_flag():
    client = MagicMock()
    orch = Orchestrator(goal="test", client=client)
    orch._done = True
    orch.tasks = []
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_no_tasks_marks_done():
    """When planner returns empty tasks, orchestrator should treat as done."""
    client = MagicMock()
    client.settings.max_cycles = 0
    client.settings.max_parallel_tasks = 5
    plan = Plan(tasks=[], rationale="nothing to do")
    planner = AsyncMock()
    planner.plan = AsyncMock(return_value=plan)

    orch = Orchestrator(goal="test", client=client)
    orch.planner = planner
    await orch.run()

    assert orch._done is True
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_output_handler_receives_messages():
    """The output handler should be called with progress messages."""
    tasks = [TaskModel(id="1", description="Create a file")]
    test_result = TestResult(passed=True, summary="ok")
    client, planner, worker_run, tester = _make_client(tasks, test_result)

    received: list[str] = []

    def handler(msg: str) -> None:
        received.append(msg)

    orch = Orchestrator(goal="test", client=client, output_handler=handler)
    orch.planner = planner
    with patch("furrow.core.orchestrator.WorkerAgent") as MockWA:
        MockWA.return_value.run = worker_run
        with patch("furrow.core.orchestrator.TesterAgent") as MockTA:
            MockTA.return_value.run = tester.run
            await orch.run()

    assert any("Furrow started" in m for m in received)
    assert any("Task 1" in m for m in received)
    assert any("Tests passed" in m for m in received)
