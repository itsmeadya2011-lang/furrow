"""Tests for the orchestrator loop logic and task tracking."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from furrow.config import Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


def make_task(task_id: str, status: str = "pending") -> TaskModel:
    return TaskModel(id=task_id, description=f"Task {task_id}", status=status)


@pytest.fixture
def orchestrator():
    return Orchestrator(goal="test goal")


def test_is_done_empty(orchestrator):
    """No tasks means immediately done."""
    assert orchestrator._is_done() is True


def test_is_done_all_completed(orchestrator):
    orchestrator._tasks = [make_task("1", "completed"), make_task("2", "completed")]
    assert orchestrator._is_done() is True


def test_is_done_with_pending(orchestrator):
    orchestrator._tasks = [make_task("1", "completed"), make_task("2", "pending")]
    assert orchestrator._is_done() is False


def test_is_done_with_failed(orchestrator):
    orchestrator._tasks = [make_task("1", "completed"), make_task("2", "failed")]
    assert orchestrator._is_done() is False


def test_get_tasks_returns_stored_tasks(orchestrator):
    tasks = [make_task("1"), make_task("2")]
    orchestrator._tasks = tasks
    assert orchestrator._get_tasks() is tasks


def test_max_cycles_zero_means_unlimited(orchestrator):
    """max_cycles=0 means unlimited (no max)."""
    assert orchestrator.max_cycles is None


def test_max_cycles_enforced():
    from furrow.config import Settings
    s = Settings(max_cycles=1)
    orch = Orchestrator(goal="test", settings=s)
    assert orch.max_cycles == 1


@pytest.mark.asyncio
async def test_max_cycles_stops_loop():
    """With max_cycles=1, the orchestrator should run one cycle and stop."""
    s = Settings(max_cycles=1)
    orch = Orchestrator(goal="test", settings=s)
    orch._cycle = AsyncMock()

    await orch.run()

    assert orch.cycles == 1
    orch._cycle.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_stops_when_done():
    """The loop should stop when all tasks are completed."""
    s = Settings(max_cycles=0)  # unlimited
    orch = Orchestrator(goal="test", settings=s)
    orch._cycle = AsyncMock()

    async def set_done():
        orch._tasks = [make_task("1", "completed")]

    orch._cycle.side_effect = set_done

    await orch.run()

    assert orch.cycles == 1
