import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_get_tasks_empty_when_no_plan():
    o = Orchestrator(goal="test")
    assert o._get_tasks() == []


def test_orchestrator_get_tasks_returns_plan_tasks():
    o = Orchestrator(goal="test")
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
        ],
        rationale="test",
    )
    o.plan = plan
    assert len(o._get_tasks()) == 2
    assert o._get_tasks()[0].id == "1"


def test_orchestrator_is_done_no_tasks():
    o = Orchestrator(goal="test")
    assert o._is_done() is True


def test_orchestrator_is_done_with_completed_tasks():
    o = Orchestrator(goal="test")
    o.plan = Plan(
        tasks=[TaskModel(id="1", description="task", status="completed")],
        rationale="test",
    )
    assert o._is_done() is True


def test_orchestrator_is_done_with_pending_tasks():
    o = Orchestrator(goal="test")
    o.plan = Plan(
        tasks=[TaskModel(id="1", description="task", status="pending")],
        rationale="test",
    )
    assert o._is_done() is False


def test_orchestrator_is_done_with_failed_tasks():
    o = Orchestrator(goal="test")
    o.plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="failed"),
        ],
        rationale="test",
    )
    assert o._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_run_halts_when_no_tasks():
    o = Orchestrator(goal="do nothing")
    o.plan = Plan(tasks=[], rationale="none")
    await o.run()
    assert o.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_max_cycles():
    from unittest.mock import AsyncMock, patch

    o = Orchestrator(goal="test")
    o.client.settings.max_cycles = 2
    with patch.object(o.planner, "plan", new_callable=AsyncMock) as mock_plan:
        mock_plan.return_value = Plan(tasks=[], rationale="none")
        await o.run()
    assert o.cycles == 2
