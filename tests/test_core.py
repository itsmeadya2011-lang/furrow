import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_get_tasks_after_plan():
    from unittest.mock import AsyncMock, MagicMock

    mock_client = MagicMock()
    mock_client.settings = MagicMock(max_parallel_tasks=5, max_cycles=0)
    mock_planner = AsyncMock()
    mock_planner.plan.return_value = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.planner = mock_planner
    orchestrator.current_plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )
    tasks = orchestrator._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].id == "1"


def test_orchestrator_is_done_true_when_all_completed():
    orchestrator = Orchestrator(goal="test")
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_false_when_pending():
    orchestrator = Orchestrator(goal="test")
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="pending"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_true_when_no_tasks():
    orchestrator = Orchestrator(goal="test")
    orchestrator.current_plan = Plan(tasks=[], rationale="ok")
    assert orchestrator._is_done() is True


def test_orchestrator_max_cycles_honored():
    from unittest.mock import AsyncMock, MagicMock

    mock_client = MagicMock()
    mock_client.settings = MagicMock(max_parallel_tasks=5, max_cycles=1)
    mock_planner = AsyncMock()
    mock_planner.plan.return_value = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.planner = mock_planner
    orchestrator.current_plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )
    assert orchestrator._is_done() is False
    assert orchestrator.settings.max_cycles == 1
