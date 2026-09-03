from pathlib import Path

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_get_tasks_empty():
    orch = Orchestrator(goal="do something")
    assert orch._get_tasks() == []


def test_orchestrator_is_done_no_plan():
    orch = Orchestrator(goal="do something")
    assert orch._is_done() is False


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert isinstance(s.workspace, Path)
    assert s.max_parallel_tasks == 5


def test_plan_multiple_tasks():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="first"),
            TaskModel(id="2", description="second"),
        ],
        rationale="r",
    )
    assert len(p.tasks) == 2
    assert p.tasks[0].id == "1"


def test_test_result_defaults():
    t = TestResult(passed=False, summary="x")
    assert t.failures == []


def test_task_model_optional_fields():
    task = TaskModel(id="1", description="do thing")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None