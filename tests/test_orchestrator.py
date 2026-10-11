from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel
from furrow.core.orchestrator import Orchestrator


def _make_task(task_id: str, status: str = "pending") -> TaskModel:
    return TaskModel(id=task_id, description=f"Task {task_id}", status=status)


def test_get_tasks_returns_internal_list():
    orch = Orchestrator(goal="test")
    orch._tasks = [_make_task("1"), _make_task("2")]
    tasks = orch._get_tasks()
    assert tasks is orch._tasks
    assert len(tasks) == 2


def test_is_done_returns_true_when_all_completed():
    orch = Orchestrator(goal="test")
    orch._tasks = [_make_task("1", "completed"), _make_task("2", "completed")]
    assert orch._is_done() is True


def test_is_done_returns_false_when_any_pending():
    orch = Orchestrator(goal="test")
    orch._tasks = [_make_task("1", "completed"), _make_task("2", "pending")]
    assert orch._is_done() is False


def test_is_done_returns_false_when_any_failed():
    orch = Orchestrator(goal="test")
    orch._tasks = [_make_task("1", "completed"), _make_task("2", "failed")]
    assert orch._is_done() is False


def test_is_done_returns_false_when_no_tasks():
    orch = Orchestrator(goal="test")
    orch._tasks = []
    assert orch._is_done() is False


async def test_cycle_merges_tasks_without_duplicates_by_id():
    orch = Orchestrator(goal="test")
    orch._tasks = [_make_task("1", "completed")]

    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[_make_task("1"), _make_task("2")],
            rationale="updated",
        )
    )
    orch.planner = mock_planner

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(
            return_value=MagicMock(passed=True, summary="ok", failures=[])
        )
        await orch._cycle()

    ids = [t.id for t in orch._get_tasks()]
    assert ids.count("1") == 1
    assert ids == ["1", "2"]


async def test_cycle_preserves_completed_status_when_planner_returns_task_again():
    orch = Orchestrator(goal="test")
    orch._tasks = [_make_task("1", "completed")]

    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[_make_task("1"), _make_task("2")],
            rationale="updated",
        )
    )
    orch.planner = mock_planner

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(
            return_value=MagicMock(passed=True, summary="ok", failures=[])
        )
        await orch._cycle()

    task1 = next(t for t in orch._get_tasks() if t.id == "1")
    assert task1.status == "completed"


async def test_cycle_preserves_failed_status_when_planner_returns_task_again():
    orch = Orchestrator(goal="test")
    orch._tasks = [_make_task("1", "failed")]

    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[_make_task("1"), _make_task("2")],
            rationale="updated",
        )
    )
    orch.planner = mock_planner

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(
            return_value=MagicMock(passed=True, summary="ok", failures=[])
        )
        await orch._cycle()

    task1 = next(t for t in orch._get_tasks() if t.id == "1")
    assert task1.status == "failed"


async def test_max_cycles_limit_halts_run():
    orch = Orchestrator(goal="test")
    orch.max_cycles = 2

    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[_make_task("1", "pending")],
            rationale="keep going",
        )
    )
    orch.planner = mock_planner

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        # Worker raises so task stays failed -> _is_done() returns False -> cycle continues
        MockWorker.return_value.run = AsyncMock(side_effect=RuntimeError("boom"))
        MockTester.return_value.run = AsyncMock(
            return_value=MagicMock(passed=True, summary="ok", failures=[])
        )
        with patch.object(orch, "_print_summary"):
            await orch.run()

    assert orch.cycles == 2