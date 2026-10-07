import pytest
from furrow.config import Plan, TaskModel, TestResult as TestResultModel, Settings
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResultModel(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_is_done_with_no_tasks():
    o = Orchestrator("test goal")
    assert o._is_done() is True


def test_orchestrator_is_done_with_completed_tasks():
    o = Orchestrator("test goal")
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert o._is_done() is True


def test_orchestrator_is_done_with_failed_tasks():
    o = Orchestrator("test goal")
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert o._is_done() is False


def test_orchestrator_max_cycles_setting():
    settings = Settings(max_cycles=3)
    assert settings.max_cycles == 3
