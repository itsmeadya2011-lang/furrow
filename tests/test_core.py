import pytest
from pathlib import Path

from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.config import Provider
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_plan_with_dependencies():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="first"),
            TaskModel(id="2", description="second", dependencies=["1"]),
        ],
        rationale="r",
    )
    assert plan.tasks[0].dependencies == []
    assert plan.tasks[1].dependencies == ["1"]


def test_task_status_default():
    t = TaskModel(id="x", description="d")
    assert t.status == "pending"


def test_settings_defaults():
    assert settings.provider == Provider.ANTHROPIC
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert isinstance(settings.workspace, Path)


def test_orchestrator_initializes():
    o = Orchestrator(goal="x")
    assert o.goal == "x"
    assert o.cycles == 0