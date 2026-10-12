"""Tests for the Orchestrator state management and lifecycle."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel
from furrow.core.orchestrator import Orchestrator


def _make_plan(tasks: list[TaskModel]) -> Plan:
    return Plan(tasks=tasks, rationale="test plan")


def _make_mock_client(max_cycles: int = 0, max_parallel: int = 5):
    settings = MagicMock()
    settings.max_cycles = max_cycles
    settings.max_parallel_tasks = max_parallel
    client = MagicMock()
    client.settings = settings
    return client


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_initial_state():
    """Orchestrator should initialize with empty task list and cycle counter."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test goal", client=client)
    assert orch.goal == "test goal"
    assert orch.cycles == 0
    assert orch.tasks == []


@pytest.mark.asyncio
async def test_orchestrator_default_client():
    """Orchestrator should create an LLMClient if none is provided."""
    with patch("furrow.core.orchestrator.LLMClient") as mock_cls:
        mock_cls.return_value = MagicMock()
        orch = Orchestrator(goal="test goal")
        assert orch.client is mock_cls.return_value
        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# _get_tasks and _is_done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_tasks_returns_persisted_tasks():
    """_get_tasks should return self.tasks, not an empty list."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    tasks = [TaskModel(id="1", description="task 1"), TaskModel(id="2", description="task 2")]
    orch.tasks = tasks
    assert orch._get_tasks() is tasks


@pytest.mark.asyncio
async def test_is_done_empty_tasks_returns_false():
    """_is_done should return False when there are no tasks (not halt immediately)."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    orch.tasks = []
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_all_completed_returns_true():
    """_is_done should return True when all tasks are completed."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    orch.tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ]
    for t in orch.tasks:
        t.status = "completed"
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_has_failures_returns_false():
    """_is_done should return False when any task has failed."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_partial_completion_returns_false():
    """_is_done should return False when some tasks are still pending."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    assert orch._is_done() is False


# ---------------------------------------------------------------------------
# _cycle: task persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cycle_persists_tasks_from_plan():
    """Tasks planned in _cycle should be stored in self.tasks for later use."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)

    plan = _make_plan([TaskModel(id="1", description="do thing")])

    with patch.object(orch.planner, "plan", new=AsyncMock(return_value=plan)):
        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            mock_worker = MagicMock()
            mock_worker.run = AsyncMock(return_value="done")
            MockWorker.return_value = mock_worker

            with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                mock_tester = MagicMock()
                mock_tester.run = AsyncMock(
                    return_value=MagicMock(passed=True, summary="ok", failures=[])
                )
                MockTester.return_value = mock_tester

                await orch._cycle()

    assert len(orch.tasks) == 1
    assert orch.tasks[0].id == "1"
    assert orch.tasks[0].status == "completed"
    assert orch.tasks[0].result == "done"


@pytest.mark.asyncio
async def test_cycle_sets_tasks_to_empty_when_no_plan_tasks():
    """When the planner returns no tasks, self.tasks should be set to empty."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    orch.tasks = [TaskModel(id="1", description="old task", status="completed")]

    plan = _make_plan([])  # empty tasks

    with patch.object(orch.planner, "plan", new=AsyncMock(return_value=plan)):
        await orch._cycle()

    assert orch.tasks == []


@pytest.mark.asyncio
async def test_cycle_marks_worker_exception_as_failed():
    """When a worker raises, its task status should be 'failed'."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)

    plan = _make_plan([TaskModel(id="1", description="do thing")])

    with patch.object(orch.planner, "plan", new=AsyncMock(return_value=plan)):
        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            mock_worker = MagicMock()
            mock_worker.run = AsyncMock(side_effect=RuntimeError("boom"))
            MockWorker.return_value = mock_worker

            with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                mock_tester = MagicMock()
                mock_tester.run = AsyncMock(
                    return_value=MagicMock(passed=True, summary="ok", failures=[])
                )
                MockTester.return_value = mock_tester

                await orch._cycle()

    assert orch.tasks[0].status == "failed"
    assert "boom" in orch.tasks[0].result


# ---------------------------------------------------------------------------
# max_cycles enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_cycles_stops_loop():
    """run() should stop after max_cycles cycles."""
    settings = MagicMock()
    settings.max_cycles = 2
    settings.max_parallel_tasks = 5
    client = MagicMock()
    client.settings = settings

    with patch("furrow.core.orchestrator.Orchestrator._cycle", new=AsyncMock()) as mock_cycle:
        # Make _is_done return False so the loop only stops via max_cycles
        with patch.object(Orchestrator, "_is_done", return_value=False):
            orch = Orchestrator(goal="test", client=client)
            await orch.run()

    assert orch.cycles == 2
    assert mock_cycle.call_count == 2


@pytest.mark.asyncio
async def test_max_cycles_zero_runs_until_done():
    """max_cycles=0 should mean infinite (run until goal complete)."""
    settings = MagicMock()
    settings.max_cycles = 0
    settings.max_parallel_tasks = 5
    client = MagicMock()
    client.settings = settings

    call_count = 0

    async def mock_cycle_and_complete_done():
        nonlocal call_count
        call_count += 1
        orch.tasks = [TaskModel(id="1", description="t", status="completed")]

    with patch.object(Orchestrator, "_cycle", side_effect=mock_cycle_and_complete_done):
        orch = Orchestrator(goal="test", client=client)
        await orch.run()

    assert call_count == 1
    assert orch.cycles == 1


# ---------------------------------------------------------------------------
# max_parallel_tasks enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_parallel_tasks_limits_concurrency():
    """Workers should be limited by max_parallel_tasks semaphore."""
    client = _make_mock_client(max_parallel=2)
    orch = Orchestrator(goal="test", client=client)

    plan = _make_plan(
        [TaskModel(id=str(i), description=f"task {i}") for i in range(4)]
    )

    current_concurrent = 0
    max_seen = 0

    async def mock_worker_run():
        nonlocal current_concurrent, max_seen
        current_concurrent += 1
        max_seen = max(max_seen, current_concurrent)
        await asyncio.sleep(0.01)
        current_concurrent -= 1
        return "done"

    with patch.object(orch.planner, "plan", new=AsyncMock(return_value=plan)):
        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            mock_worker = MagicMock()
            mock_worker.run = mock_worker_run
            MockWorker.return_value = mock_worker

            with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                mock_tester = MagicMock()
                mock_tester.run = AsyncMock(
                    return_value=MagicMock(passed=True, summary="ok", failures=[])
                )
                MockTester.return_value = mock_tester

                await orch._cycle()

    assert max_seen <= 2

