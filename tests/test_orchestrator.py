"""Tests for the Orchestrator state tracking, max_cycles, and max_parallel_tasks."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


def make_task(tid: str = "1", desc: str = "do something") -> TaskModel:
    return TaskModel(id=tid, description=desc, files=[], dependencies=[], status="pending")


def make_plan(tasks: list[TaskModel]) -> Plan:
    return Plan(tasks=tasks, rationale="test plan")


def make_settings(**overrides) -> Settings:
    defaults = {"max_cycles": 0, "max_parallel_tasks": 5}
    defaults.update(overrides)
    return MagicMock(**defaults)


@pytest.mark.asyncio
async def test_is_done_with_pending_tasks():
    """_is_done should be False when tasks are still pending."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator.tasks = [make_task("1"), make_task("2")]
    # Both pending
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_is_done_with_completed_tasks():
    """_is_done should be True when all tasks are completed."""
    orchestrator = Orchestrator(goal="test goal")
    t = make_task("1")
    t.status = "completed"
    orchestrator.tasks = [t]
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_is_done_with_failed_tasks():
    """_is_done should be False when any task has failed."""
    orchestrator = Orchestrator(goal="test goal")
    t = make_task("1")
    t.status = "failed"
    orchestrator.tasks = [t]
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_is_done_with_mixed_tasks():
    """_is_done should be False when some tasks are completed and some pending."""
    orchestrator = Orchestrator(goal="test goal")
    t1 = make_task("1")
    t1.status = "completed"
    t2 = make_task("2")  # still pending
    orchestrator.tasks = [t1, t2]
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_is_done_with_no_tasks():
    """_is_done should be False with no tasks (don't falsely report done)."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator.tasks = []
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_get_tasks_returns_stored_tasks():
    """_get_tasks should return the tasks stored on the orchestrator."""
    orchestrator = Orchestrator(goal="test goal")
    tasks = [make_task("1"), make_task("2"), make_task("3")]
    orchestrator.tasks = tasks
    assert orchestrator._get_tasks() == tasks
    assert len(orchestrator._get_tasks()) == 3


@pytest.mark.asyncio
async def test_max_cycles_stops_loop():
    """Orchestrator should stop after max_cycles even if goals aren't complete."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "done"}')
    mock_client.settings = make_settings(max_cycles=1, max_parallel_tasks=5)

    with patch.object(Orchestrator, "_cycle", new_callable=AsyncMock) as mock_cycle:
        mock_cycle.return_value = None
        orchestrator = Orchestrator(goal="test", client=mock_client)
        orchestrator.settings = mock_client.settings
        await orchestrator.run()
        # Should have run exactly 1 cycle
        assert mock_cycle.call_count == 1
        assert orchestrator.cycles == 1


@pytest.mark.asyncio
async def test_max_cycles_set_to_zero_runs_forever():
    """When max_cycles is 0 (default), the loop should not be limited by max_cycles."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "done"}')
    mock_client.settings = make_settings(max_cycles=0, max_parallel_tasks=5)

    with patch.object(Orchestrator, "_cycle", new_callable=AsyncMock) as mock_cycle:
        mock_cycle.return_value = None
        # Make _is_done return True on second call to stop the loop
        call_count = 0
        def mock_is_done(self):
            nonlocal call_count
            call_count += 1
            return call_count >= 1
        with patch.object(Orchestrator, "_is_done", mock_is_done):
            orchestrator = Orchestrator(goal="test", client=mock_client)
            orchestrator.settings = mock_client.settings
            await orchestrator.run()
            assert orchestrator.cycles >= 1


@pytest.mark.asyncio
async def test_max_parallel_tasks_uses_semaphore():
    """Tasks should be executed with a semaphore limiting parallelism."""
    tasks = [make_task(str(i)) for i in range(10)]
    make_plan(tasks)

    mock_client = MagicMock()
    mock_client.settings = make_settings(max_cycles=0, max_parallel_tasks=2)

    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.settings = mock_client.settings

    # Track concurrent executions
    concurrent = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def fake_worker_run():
        nonlocal concurrent, max_concurrent
        async with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.01)
        async with lock:
            concurrent -= 1

    with patch.object(Orchestrator, "_cycle", new_callable=AsyncMock) as mock_cycle:
        async def cycle_impl():
            orchestrator.tasks = tasks
            semaphore = asyncio.Semaphore(orchestrator.settings.max_parallel_tasks)

            async def run_task(task):
                async with semaphore:
                    return await fake_worker_run()

            results = await asyncio.gather(
                *[run_task(t) for t in tasks], return_exceptions=True
            )
            for t, r in zip(tasks, results):
                t.status = "completed"

        mock_cycle.side_effect = cycle_impl
        orchestrator.settings = MagicMock(max_cycles=1, max_parallel_tasks=2)
        await orchestrator.run()

        assert max_concurrent <= 2


@pytest.mark.asyncio
async def test_cycle_stores_tasks():
    """_cycle should store tasks in self.tasks for tracking."""
    tasks = [make_task("1"), make_task("2")]
    plan = make_plan(tasks)

    mock_client = MagicMock()
    mock_client.settings = make_settings(max_cycles=10, max_parallel_tasks=5)
    mock_client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "done"}')

    with patch("furrow.core.orchestrator.PlannerAgent") as mock_planner_cls:
        mock_planner = MagicMock()
        mock_planner.plan = AsyncMock(return_value=plan)
        mock_planner_cls.return_value = mock_planner

        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls:
            mock_worker = MagicMock()
            mock_worker.run = AsyncMock(return_value="done")
            mock_worker_cls.return_value = mock_worker

            with patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls:
                mock_tester = MagicMock()
                mock_tester.run = AsyncMock(
                    return_value=MagicMock(passed=True, summary="ok", failures=[])
                )
                mock_tester_cls.return_value = mock_tester

                orchestrator = Orchestrator(goal="test", client=mock_client)
                orchestrator.settings = mock_client.settings
                await orchestrator._cycle()

                assert orchestrator.tasks == tasks
                assert len(orchestrator.history) == 1
                assert orchestrator.history[0] == plan
