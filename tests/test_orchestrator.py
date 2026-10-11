from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


def _make_task(task_id="1", description="do thing", status="pending"):
    return TaskModel(id=task_id, description=description, status=status)


def test_is_done_all_completed():
    orch = Orchestrator(goal="test")
    orch.tasks = [_make_task("1", status="completed"), _make_task("2", status="completed")]
    assert orch._is_done() is True


def test_is_done_any_failed():
    orch = Orchestrator(goal="test")
    orch.tasks = [_make_task("1", status="completed"), _make_task("2", status="failed")]
    assert orch._is_done() is False


def test_is_done_no_tasks():
    orch = Orchestrator(goal="test")
    orch.tasks = []
    assert orch._is_done() is True


def test_is_done_max_cycles_zero_means_unlimited():
    """max_cycles == 0 means no limit."""
    orch = Orchestrator(goal="test", settings=Settings(max_cycles=0))
    orch.tasks = [_make_task("1", status="pending")]
    orch.cycles = 100
    assert orch._is_done() is False


def test_is_done_respects_max_cycles():
    orch = Orchestrator(goal="test", settings=Settings(max_cycles=3))
    orch.tasks = [_make_task("1", status="pending")]
    orch.cycles = 3
    assert orch._is_done() is True


def test_is_done_max_cycles_not_reached_yet():
    orch = Orchestrator(goal="test", settings=Settings(max_cycles=5))
    orch.tasks = [_make_task("1", status="pending")]
    orch.cycles = 2
    assert orch._is_done() is False


def test_get_tasks_returns_stored_tasks():
    orch = Orchestrator(goal="test")
    tasks = [_make_task("1"), _make_task("2")]
    orch.tasks = tasks
    assert orch._get_tasks() is tasks


def test_get_tasks_empty_by_default():
    orch = Orchestrator(goal="test")
    assert orch._get_tasks() == []


@pytest.mark.asyncio
async def test_cycle_stores_tasks_from_plan():
    """Test that tasks are stored after _cycle() runs (mock the planner)."""
    orch = Orchestrator(goal="test")

    plan = Plan(
        tasks=[TaskModel(id="1", description="task one"), TaskModel(id="2", description="task two")],
        rationale="split it",
    )

    # Mock the planner
    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(return_value=plan)
    orch.planner = mock_planner

    # Mock the worker and tester to avoid real LLM calls
    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
        mock_worker_instance = MagicMock()
        mock_worker_instance.run = AsyncMock(return_value="done")
        MockWorker.return_value = mock_worker_instance

        with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            mock_tester_instance = MagicMock()
            mock_tester_instance.run = AsyncMock(return_value=MagicMock(passed=True, summary="ok", failures=[]))
            MockTester.return_value = mock_tester_instance

            await orch._cycle()

    assert len(orch.tasks) == 2
    assert orch.tasks[0].id == "1"
    assert orch.tasks[1].id == "2"
    assert orch.tasks[0].status == "completed"
    assert orch.tasks[1].status == "completed"


@pytest.mark.asyncio
async def test_cycle_with_no_tasks():
    """When planner returns empty plan, tasks should be empty."""
    orch = Orchestrator(goal="test")

    plan = Plan(tasks=[], rationale="nothing to do")
    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(return_value=plan)
    orch.planner = mock_planner

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
        with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            await orch._cycle()

    assert orch.tasks == []
    # Workers should not have been instantiated
    MockWorker.assert_not_called()


@pytest.mark.asyncio
async def test_cycle_marks_failed_tasks():
    """Tasks that raise exceptions should be marked as failed."""
    orch = Orchestrator(goal="test")

    plan = Plan(
        tasks=[TaskModel(id="1", description="fail task")],
        rationale="test failure",
    )
    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(return_value=plan)
    orch.planner = mock_planner

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
        mock_worker_instance = MagicMock()
        mock_worker_instance.run = AsyncMock(side_effect=RuntimeError("boom"))
        MockWorker.return_value = mock_worker_instance

        with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            mock_tester_instance = MagicMock()
            mock_tester_instance.run = AsyncMock(return_value=MagicMock(passed=False, summary="fail", failures=["boom"]))
            MockTester.return_value = mock_tester_instance

            await orch._cycle()

    assert orch.tasks[0].status == "failed"
    assert "boom" in orch.tasks[0].result