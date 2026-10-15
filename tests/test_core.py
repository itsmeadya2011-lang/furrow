"""Tests for config models and TaskModel behavior."""
from __future__ import annotations

from furrow.config import Plan, Provider, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_plan_with_files_and_dependencies():
    """Plan correctly stores files and dependencies on tasks."""
    p = Plan(
        tasks=[
            TaskModel(
                id="1",
                description="Implement auth",
                files=["src/auth.py", "tests/test_auth.py"],
                dependencies=[],
            ),
            TaskModel(
                id="2",
                description="Add login page",
                files=["src/login.py"],
                dependencies=["1"],
            ),
        ],
        rationale="Auth then login",
    )
    assert len(p.tasks) == 2
    assert p.tasks[0].files == ["src/auth.py", "tests/test_auth.py"]
    assert p.tasks[1].dependencies == ["1"]


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_with_failures():
    """TestResult stores failures list."""
    t = TestResult(passed=False, summary="3 failures", failures=["a == b", "c != d"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_test_result_defaults():
    """TestResult defaults failures to empty list."""
    t = TestResult(passed=True, summary="ok")
    assert t.failures == []


def test_task_model_defaults():
    """TaskModel defaults files and dependencies to empty lists."""
    t = TaskModel(id="1", description="task")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_provider_enum():
    """Provider enum has expected values."""
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_plan_json_roundtrip():
    """Plan can be serialized and deserialized."""
    original = Plan(
        tasks=[TaskModel(id="1", description="task 1", files=["f.py"], dependencies=["2"])],
        rationale="test",
    )
    data = original.model_dump()
    restored = Plan(**data)
    assert restored.tasks[0].description == "task 1"
    assert restored.tasks[0].files == ["f.py"]
    assert restored.tasks[0].dependencies == ["2"]
    assert restored.rationale == "test"
