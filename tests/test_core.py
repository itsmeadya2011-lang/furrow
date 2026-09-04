import pytest
from unittest.mock import AsyncMock, patch
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_get_tasks_returns_plan_tasks():
    orch = Orchestrator("test goal")
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task one"),
            TaskModel(id="2", description="task two"),
        ],
        rationale="test plan",
    )
    orch._latest_plan = plan
    tasks = orch._get_tasks()
    assert len(tasks) == 2
    assert tasks[0].id == "1"
    assert tasks[1].id == "2"


def test_orchestrator_is_done_no_plan():
    orch = Orchestrator("test goal")
    assert orch._is_done() is False


def test_orchestrator_is_done_all_completed():
    orch = Orchestrator("test goal")
    orch._latest_plan = Plan(
        tasks=[
            TaskModel(id="1", description="task one", status="completed"),
            TaskModel(id="2", description="task two", status="completed"),
        ],
        rationale="test plan",
    )
    assert orch._is_done() is True


def test_orchestrator_is_done_with_failure():
    orch = Orchestrator("test goal")
    orch._latest_plan = Plan(
        tasks=[
            TaskModel(id="1", description="task one", status="completed"),
            TaskModel(id="2", description="task two", status="failed"),
        ],
        rationale="test plan",
    )
    assert orch._is_done() is False


def test_orchestrator_is_done_empty_tasks():
    orch = Orchestrator("test goal")
    orch._latest_plan = Plan(tasks=[], rationale="test plan")
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_max_cycles_enforced():
    with patch('furrow.core.orchestrator.settings') as mock_settings:
        mock_settings.max_cycles = 2
        orch = Orchestrator("test goal")
        mock_cycle = AsyncMock()
        with patch.object(orch, '_cycle', mock_cycle):
            await orch.run()
        assert orch.cycles == 2


@pytest.mark.asyncio
async def test_orchestrator_keyboard_interrupt():
    orch = Orchestrator("test goal")
    mock_cycle = AsyncMock(side_effect=KeyboardInterrupt)
    with patch.object(orch, '_cycle', mock_cycle):
        await orch.run()
    assert orch.cycles == 1
