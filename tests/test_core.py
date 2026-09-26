"""Tests for core data models (Plan, TaskModel, TestResult)."""
from __future__ import annotations

from furrow.config import Plan, TaskModel, TestResult


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.id == "1"
    assert t.description == "do thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_values():
    t = TaskModel(
        id="2",
        description="write code",
        files=["src/a.py", "src/b.py"],
        dependencies=["1"],
        status="completed",
        result="done",
    )
    assert t.files == ["src/a.py", "src/b.py"]
    assert t.dependencies == ["1"]
    assert t.status == "completed"
    assert t.result == "done"


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"


def test_plan_from_dict():
    data = {
        "tasks": [
            {
                "id": "1",
                "description": "Add logging",
                "files": ["src/log.py"],
                "dependencies": [],
            }
        ],
        "rationale": "Need to add structured logging",
    }
    p = Plan(**data)
    assert len(p.tasks) == 1
    assert p.tasks[0].id == "1"
    assert p.tasks[0].files == ["src/log.py"]


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="Nothing to do")
    assert p.tasks == []


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
    assert t.summary == "ok"
    assert t.failures == []


def test_test_result_with_failures():
    t = TestResult(
        passed=False,
        summary="2 tests failed",
        failures=["test_a.py::test_1 failed: assertion", "test_b.py::test_2 failed: timeout"],
    )
    assert t.passed is False
    assert len(t.failures) == 2
    assert "timeout" in t.failures[1]


def test_test_result_defaults():
    t = TestResult(passed=True, summary="All good")
    assert t.failures == []
