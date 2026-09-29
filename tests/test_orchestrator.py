"""Tests for the Orchestrator's task scheduling and lifecycle logic.

These tests use mock/fake agents so they run without any external API calls.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _make_plan(tasks_data=None) -> Plan:
    if tasks_data is None:
        tasks_data = []
    tasks = [
        TaskModel(id=t["id"], description=t["description"], files=t.get("files", []),
                  dependencies=t.get("dependencies", []))
        for t in tasks_data
    ]
    return Plan(tasks=tasks, rationale="test plan")


# --------------------------------------------------------------------------- #
# _is_done / _get_tasks
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_tasks_empty_when_no_plan():
    orch = Orchestrator(goal="do something")
    assert orch._get_tasks() == []


@pytest.mark.asyncio
async def test_is_done_true_when_no_plan():
    orch = Orchestrator(goal="do something")
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_true_when_all_completed():
    orch = Orchestrator(goal="do something")
    orch._plan = _make_plan([{"id": "1", "description": "task1"}, {"id": "2", "description": "task2"}])
    for t in orch._plan.tasks:
        t.status = "completed"
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_false_when_any_failed():
    orch = Orchestrator(goal="do something")
    orch._plan = _make_plan([{"id": "1", "description": "task1"}, {"id": "2", "description": "task2"}])
    orch._plan.tasks[0].status = "completed"
    orch._plan.tasks[1].status = "failed"
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_false_when_some_pending():
    orch = Orchestrator(goal="do something")
    orch._plan = _make_plan([{"id": "1", "description": "task1"}, {"id": "2", "description": "task2"}])
    orch._plan.tasks[0].status = "completed"
    orch._plan.tasks[1].status = "pending"
    assert orch._is_done() is False


# --------------------------------------------------------------------------- #
# max_cycles enforcement
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_max_cycles_halts():
    """With max_cycles=2, the orchestrator should stop after 2 cycles."""
    with patch("furrow.core.orchestrator.PlannerAgent"), \
         patch("furrow.core.orchestrator.WorkerAgent"), \
         patch("furrow.core.orchestrator.TesterAgent"):
        orch = Orchestrator(goal="something")
        orch.max_cycles = 2

        async def mock_cycle():
            if orch._plan is None:
                orch._plan = _make_plan([{"id": "1", "description": "x"}])
                orch._plan.tasks[0].status = "pending"

        orch._cycle = mock_cycle  # type: ignore[assignment]
        await orch.run()
        assert orch.cycles == 2


@pytest.mark.asyncio
async def test_zero_max_cycles_means_unlimited():
    """With max_cycles=0, the orchestrator should run until _is_done() returns True."""
    with patch("furrow.core.orchestrator.PlannerAgent"), \
         patch("furrow.core.orchestrator.TesterAgent"):
        orch = Orchestrator(goal="something")
        orch.max_cycles = 0
        call_count = 0

        async def mock_cycle():
            nonlocal call_count
            call_count += 1
            if orch._plan is None:
                orch._plan = _make_plan([{"id": "1", "description": "x"}])
                orch._plan.tasks[0].status = "pending"
            else:
                orch._plan.tasks[0].status = "completed"

        orch._cycle = mock_cycle  # type: ignore[assignment]
        await orch.run()
        assert call_count == 2


# --------------------------------------------------------------------------- #
# Dependency-aware scheduling (no LLM calls)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_scheduling_runs_independent_tasks():
    """Tasks with no dependencies should all run in the same wave."""
    with patch("furrow.core.orchestrator.PlannerAgent"), \
         patch("furrow.core.orchestrator.TesterAgent"):
        orch = Orchestrator(goal="test")
        orch.max_parallel = 5

        tasks = _make_plan([
            {"id": "1", "description": "a"},
            {"id": "2", "description": "b"},
        ])

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            instance = MockWorker.return_value
            instance.run = AsyncMock(side_effect=["result1", "result2"])
            results = await orch._run_scheduled_tasks(tasks.tasks)

    assert results == ["result1", "result2"]
    assert MockWorker.call_count == 2


@pytest.mark.asyncio
async def test_scheduling_respects_dependency_order():
    """Task 2 depends on task 1, so task 1 must complete before task 2 runs."""
    with patch("furrow.core.orchestrator.PlannerAgent"), \
         patch("furrow.core.orchestrator.TesterAgent"):
        orch = Orchestrator(goal="test")
        orch.max_parallel = 5

        tasks = _make_plan([
            {"id": "1", "description": "a"},
            {"id": "2", "description": "b", "dependencies": ["1"]},
        ])

        call_order = []

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            def make_worker(task, **kwargs):
                mock = MagicMock()
                async def _run():
                    call_order.append(task.id)
                    return f"result-{task.id}"
                mock.run = _run
                return mock

            MockWorker.side_effect = make_worker
            results = await orch._run_scheduled_tasks(tasks.tasks)

    assert call_order == ["1", "2"]
    assert results == ["result-1", "result-2"]


@pytest.mark.asyncio
async def test_scheduling_skips_when_dependency_fails():
    """If a task fails, dependent tasks should be skipped."""
    with patch("furrow.core.orchestrator.PlannerAgent"), \
         patch("furrow.core.orchestrator.TesterAgent"):
        orch = Orchestrator(goal="test")
        orch.max_parallel = 5

        tasks = _make_plan([
            {"id": "1", "description": "a"},
            {"id": "2", "description": "b", "dependencies": ["1"]},
            {"id": "3", "description": "c", "dependencies": ["1"]},
        ])

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            MockWorker.return_value.run = AsyncMock(side_effect=RuntimeError("boom"))
            results = await orch._run_scheduled_tasks(tasks.tasks)

    assert isinstance(results[0], RuntimeError)
    assert "Skipped" in results[1]
    assert "Skipped" in results[2]


@pytest.mark.asyncio
async def test_scheduling_handles_dependency_cycle():
    """Circular dependencies should not cause infinite loops."""
    with patch("furrow.core.orchestrator.PlannerAgent"), \
         patch("furrow.core.orchestrator.TesterAgent"):
        orch = Orchestrator(goal="test")
        orch.max_parallel = 5

        tasks = _make_plan([
            {"id": "1", "description": "a", "dependencies": ["2"]},
            {"id": "2", "description": "b", "dependencies": ["1"]},
        ])

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            results = await orch._run_scheduled_tasks(tasks.tasks)

    assert "Skipped" in results[0]
    assert "Skipped" in results[1]


@pytest.mark.asyncio
async def test_max_parallel_limits_concurrency():
    """The semaphore should cap concurrent task execution."""
    with patch("furrow.core.orchestrator.PlannerAgent"), \
         patch("furrow.core.orchestrator.TesterAgent"):
        orch = Orchestrator(goal="test")
        orch.max_parallel = 2

        tasks = _make_plan([
            {"id": str(i), "description": f"task-{i}"} for i in range(6)
        ])

        current = 0
        max_observed = 0

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            def make_worker(task, **kwargs):
                mock = MagicMock()
                async def _run():
                    nonlocal current, max_observed
                    current += 1
                    max_observed = max(max_observed, current)
                    await asyncio.sleep(0.01)
                    current -= 1
                    return "ok"
                mock.run = _run
                return mock

            MockWorker.side_effect = make_worker
            await orch._run_scheduled_tasks(tasks.tasks)

    assert max_observed <= orch.max_parallel
    assert max_observed == 2


# --------------------------------------------------------------------------- #
# Full run() lifecycle (mocked agents)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_happy_path_single_cycle():
    """A single successful cycle should complete and halt."""
    mock_plan = _make_plan([{"id": "1", "description": "write file"}])

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
         patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockPlanner.return_value.plan = AsyncMock(return_value=mock_plan)
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="all pass", failures=[])
        )

        orch = Orchestrator(goal="make a file")
        result = await orch.run()

    assert result is not None
    assert orch.cycles == 1
    assert orch._plan.tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_run_two_cycles_until_done():
    """First cycle fails, second cycle succeeds → 2 cycles total."""
    plan1 = _make_plan([{"id": "1", "description": "task"}])
    plan2 = _make_plan([{"id": "2", "description": "fix task"}])

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
         patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockPlanner.return_value.plan = AsyncMock(side_effect=[plan1, plan2])
        # First call raises (task fails), second call succeeds
        MockWorker.return_value.run = AsyncMock(side_effect=[RuntimeError("task failed"), "done"])
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="all pass", failures=[])
        )

        orch = Orchestrator(goal="fix things")
        await orch.run()

    assert orch.cycles == 2