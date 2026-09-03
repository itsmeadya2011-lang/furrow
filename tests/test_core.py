import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_get_tasks_returns_plan_tasks():
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.original_goal = "test goal"
    orchestrator.goal = "test goal"
    orchestrator.cycles = 0
    orchestrator.last_plan = plan
    orchestrator.client = None
    tasks = orchestrator._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].id == "1"


def test_orchestrator_is_done_false_when_tasks_pending():
    plan = Plan(tasks=[TaskModel(id="1", description="do thing", status="pending")], rationale="ok")
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.original_goal = "test goal"
    orchestrator.goal = "test goal"
    orchestrator.cycles = 0
    orchestrator.last_plan = plan
    orchestrator.client = None
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_false_when_all_tasks_completed():
    plan = Plan(tasks=[TaskModel(id="1", description="do thing", status="completed")], rationale="ok")
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.original_goal = "test goal"
    orchestrator.goal = "test goal"
    orchestrator.cycles = 0
    orchestrator.last_plan = plan
    orchestrator.client = None
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_true_when_no_tasks():
    plan = Plan(tasks=[], rationale="nothing to do")
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.original_goal = "test goal"
    orchestrator.goal = "test goal"
    orchestrator.cycles = 0
    orchestrator.last_plan = plan
    orchestrator.client = None
    assert orchestrator._is_done() is True
