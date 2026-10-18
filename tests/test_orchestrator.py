import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import TaskModel
from furrow.core.orchestrator import Orchestrator


def make_orchestrator(tasks=None):
    client = MagicMock()
    client.settings.max_cycles = 0
    client.settings.max_parallel_tasks = 2
    orch = Orchestrator(goal="goal", client=client)
    if tasks is not None:
        orch.current_tasks = tasks
    return orch, client


def test_is_done_empty():
    orch, _ = make_orchestrator([])
    assert orch._is_done() is False


def test_is_done_all_completed():
    tasks = [TaskModel(id="1", description="a", status="completed")]
    orch, _ = make_orchestrator(tasks)
    assert orch._is_done() is True


def test_is_done_failed():
    tasks = [TaskModel(id="1", description="a", status="failed")]
    orch, _ = make_orchestrator(tasks)
    assert orch._is_done() is False


def test_is_done_mixed():
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    orch, _ = make_orchestrator(tasks)
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_current_tasks_set_after_cycle():
    client = MagicMock()
    client.settings.max_cycles = 0
    client.settings.max_parallel_tasks = 2
    orch = Orchestrator(goal="goal", client=client)
    plan = MagicMock()
    plan.tasks = [TaskModel(id="1", description="a")]
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)
    tester = MagicMock()
    tester.run = AsyncMock(return_value=MagicMock(passed=True, summary="ok", failures=[]))
    with patch("furrow.core.orchestrator.TesterAgent", return_value=tester), \
         patch("furrow.core.orchestrator.WorkerAgent") as wa:
        wa.return_value.run = AsyncMock(return_value="done")
        await orch._cycle()
    assert orch.current_tasks == plan.tasks


@pytest.mark.asyncio
async def test_max_cycles_enforcement():
    client = MagicMock()
    client.settings.max_cycles = 1
    client.settings.max_parallel_tasks = 2
    orch = Orchestrator(goal="goal", client=client)
    plan = MagicMock()
    plan.tasks = [TaskModel(id="1", description="a")]
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)
    tester = MagicMock()
    tester.run = AsyncMock(return_value=MagicMock(passed=True, summary="ok", failures=[]))
    with patch("furrow.core.orchestrator.TesterAgent", return_value=tester), \
         patch("furrow.core.orchestrator.WorkerAgent") as wa:
        wa.return_value.run = AsyncMock(return_value="done")
        await orch.run()
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_semaphore_usage():
    client = MagicMock()
    client.settings.max_cycles = 1
    client.settings.max_parallel_tasks = 1
    orch = Orchestrator(goal="goal", client=client)
    plan = MagicMock()
    plan.tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)
    tester = MagicMock()
    tester.run = AsyncMock(return_value=MagicMock(passed=True, summary="ok", failures=[]))
    with patch("furrow.core.orchestrator.TesterAgent", return_value=tester), \
         patch("furrow.core.orchestrator.WorkerAgent") as wa:
        wa.return_value.run = AsyncMock(return_value="done")
        await orch._cycle()
    assert wa.call_count == 2
