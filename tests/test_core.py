import os

import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.status == "pending"
    assert t.result is None
    assert t.files == []
    assert t.dependencies == []


def test_plan_multiple_tasks():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="task one", dependencies=[]),
            TaskModel(id="2", description="task two", dependencies=["1"]),
        ],
        rationale="sequential",
    )
    assert len(p.tasks) == 2
    assert p.tasks[1].dependencies == ["1"]
