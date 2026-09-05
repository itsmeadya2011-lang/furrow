from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_orchestrator_init():
    orch = Orchestrator(goal="test goal")
    assert orch.goal == "test goal"
    assert orch.cycles == 0


def test_orchestrator_get_tasks_with_plan():
    orch = Orchestrator(goal="test")
    orch.plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok")
    tasks = orch._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].id == "1"


def test_orchestrator_get_tasks_without_plan():
    orch = Orchestrator(goal="test")
    tasks = orch._get_tasks()
    assert tasks == []


def test_orchestrator_is_done_completed():
    orch = Orchestrator(goal="test")
    orch.plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok")
    orch.plan.tasks[0].status = "completed"
    assert orch._is_done() is True


def test_orchestrator_is_done_pending():
    orch = Orchestrator(goal="test")
    orch.plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok")
    assert orch._is_done() is False


def test_orchestrator_is_done_failed():
    orch = Orchestrator(goal="test")
    orch.plan = Plan(
        tasks=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")],
        rationale="ok",
    )
    orch.plan.tasks[0].status = "completed"
    orch.plan.tasks[1].status = "failed"
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_cycle_no_tasks():
    orch = Orchestrator(goal="test")
    mock_planner = AsyncMock()
    mock_planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="none"))
    orch.planner = mock_planner
    await orch._cycle()


@pytest.mark.asyncio
async def test_orchestrator_full_run_stops_on_success():
    orch = Orchestrator(goal="test")
    mock_plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok")

    mock_planner = AsyncMock()
    mock_planner.plan = AsyncMock(return_value=mock_plan)
    orch.planner = mock_planner

    mock_worker_instance = MagicMock()
    mock_worker_instance.run = AsyncMock(return_value="done")

    mock_tester_instance = MagicMock()
    mock_tester_instance.run = AsyncMock(
        return_value=TestResult(passed=True, summary="ok", failures=[])
    )

    with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker_instance), \
         patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester_instance):
        with patch("furrow.core.orchestrator.console"):
            await orch.run()

    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_full_run_retries_on_failure():
    orch = Orchestrator(goal="test")

    mock_plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok")

    mock_planner = AsyncMock()
    mock_planner.plan = AsyncMock(return_value=mock_plan)
    orch.planner = mock_planner

    mock_worker_instance = MagicMock()
    mock_worker_instance.run = AsyncMock(return_value="done")

    mock_tester_instance = MagicMock()
    mock_tester_instance.run = AsyncMock(
        return_value=TestResult(passed=False, summary="fail", failures=["err"])
    )

    with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker_instance), \
         patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester_instance):
        with patch("furrow.core.orchestrator.console"):
            await orch.run()

    assert orch.cycles == 1
    assert "Fix failing tests" in orch.goal
