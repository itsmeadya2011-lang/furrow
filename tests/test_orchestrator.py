from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _make_plan(tasks: list[TaskModel]) -> Plan:
    return Plan(tasks=tasks, rationale="ok")


async def test_task_storage(mock_client) -> None:
    orch = Orchestrator(goal="do something", client=mock_client)
    plan = _make_plan(
        [TaskModel(id="1", description="task 1"), TaskModel(id="2", description="task 2")]
    )
    orch.planner = MagicMock(plan=AsyncMock(return_value=plan))

    with (
        patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
        patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
    ):
        mock_worker_cls.return_value.run = AsyncMock(return_value="task result")
        mock_tester_cls.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok")
        )
        await orch._cycle()

    assert len(orch.tasks) == 2
    assert all(t.status == "completed" for t in orch.tasks)
    assert all(t.result is not None for t in orch.tasks)


def test_is_done_complete(mock_client) -> None:
    orch = Orchestrator(goal="x", client=mock_client)
    orch.tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_with_failures(mock_client) -> None:
    orch = Orchestrator(goal="x", client=mock_client)
    orch.tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="failed"),
    ]
    assert orch._is_done() is False


def test_is_done_no_tasks(mock_client) -> None:
    orch = Orchestrator(goal="x", client=mock_client)
    orch.tasks = []
    assert orch._is_done() is False


async def test_max_cycles_enforcement(mock_client) -> None:
    mock_client.settings.max_cycles = 3
    orch = Orchestrator(goal="x", client=mock_client)
    orch.planner = MagicMock(plan=AsyncMock(return_value=_make_plan([])))

    with (
        patch("furrow.core.orchestrator.WorkerAgent"),
        patch("furrow.core.orchestrator.TesterAgent"),
    ):
        await orch.run()

    assert orch.cycles == 3


async def test_no_max_cycles_infinite(mock_client) -> None:
    mock_client.settings.max_cycles = 0
    orch = Orchestrator(goal="x", client=mock_client)
    plan = _make_plan([TaskModel(id="1", description="t1")])
    orch.planner = MagicMock(plan=AsyncMock(return_value=plan))

    with (
        patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
        patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
    ):
        mock_worker_cls.return_value.run = AsyncMock(return_value="done")
        mock_tester_cls.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok")
        )
        await orch.run()

    assert orch.cycles == 1
    assert orch._is_done() is True


def test_get_tasks_returns_stored(mock_client) -> None:
    orch = Orchestrator(goal="x", client=mock_client)
    tasks = [TaskModel(id="1", description="t1")]
    orch.tasks = tasks
    assert orch._get_tasks() is tasks
    assert orch._get_tasks() == tasks


async def test_on_progress_callback(mock_client) -> None:
    received: list[str] = []
    orch = Orchestrator(goal="x", client=mock_client, on_progress=received.append)
    plan = _make_plan([TaskModel(id="1", description="t1")])
    orch.planner = MagicMock(plan=AsyncMock(return_value=plan))

    with (
        patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
        patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
    ):
        mock_worker_cls.return_value.run = AsyncMock(return_value="done")
        mock_tester_cls.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok")
        )
        await orch._cycle()

    assert len(received) > 0
    assert "Planning..." in received
