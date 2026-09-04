import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_status_defaults():
    task = TaskModel(id="1", description="do thing")
    assert task.status == "pending"
    assert task.result is None


def test_plan_with_no_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert p.tasks == []


def test_orchestrator_is_done_empty():
    from furrow.core.orchestrator import Orchestrator
    from unittest.mock import MagicMock
    orch = Orchestrator.__new__(Orchestrator)
    orch.current_plan = None
    assert orch._is_done() is False


def test_orchestrator_is_done_all_completed():
    from furrow.core.orchestrator import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)
    orch.current_plan = Plan(
        tasks=[TaskModel(id="1", description="a", status="completed")],
        rationale="done",
    )
    assert orch._is_done() is True


def test_orchestrator_is_done_with_failure():
    from furrow.core.orchestrator import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)
    orch.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ],
        rationale="fail",
    )
    assert orch._is_done() is False
