"""Tests for config models and data structures."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from furrow.config import (
    FileOperation,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
)


# --- Provider ---


def test_provider_enum_values():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_provider_is_str_enum():
    """Provider should be a str Enum so it serializes correctly in configs."""
    assert isinstance(Provider.OLLAMA, str)
    assert Provider.OLLAMA.value == "ollama"


# --- TaskModel ---


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.id == "1"
    assert t.description == "do thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_all_fields():
    t = TaskModel(
        id="task-2",
        description="Add auth",
        files=["src/auth.py", "tests/test_auth.py"],
        dependencies=["task-1"],
        status="completed",
        result="Done",
    )
    assert t.id == "task-2"
    assert t.files == ["src/auth.py", "tests/test_auth.py"]
    assert t.dependencies == ["task-1"]
    assert t.status == "completed"
    assert t.result == "Done"


def test_task_model_invalid_id_missing():
    with pytest.raises(ValidationError):
        TaskModel(description="no id")


def test_task_model_status_transitions():
    """Verify default status and that it can be changed."""
    t = TaskModel(id="1", description="x")
    assert t.status == "pending"
    t.status = "completed"
    assert t.status == "completed"
    t.status = "failed"
    assert t.status == "failed"


# --- FileOperation ---


def test_file_operation_write():
    op = FileOperation(operation="write", path="src/foo.py", content="print('hi')")
    assert op.operation == "write"
    assert op.path == "src/foo.py"
    assert op.content == "print('hi')"
    assert op.old is None
    assert op.new is None


def test_file_operation_edit():
    op = FileOperation(operation="edit", path="src/foo.py", old="hi", new="bye")
    assert op.operation == "edit"
    assert op.old == "hi"
    assert op.new == "bye"
    assert op.content is None


def test_file_operation_minimal():
    op = FileOperation(operation="write", path="x.py")
    assert op.content is None
    assert op.old is None
    assert op.new is None


# --- Plan ---


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert p.tasks == []
    assert p.rationale == "nothing to do"


def test_plan_from_dict():
    """Simulate what PlannerAgent does: Plan(**data)."""
    data = {
        "tasks": [
            {"id": "1", "description": "task1", "files": ["a.py"], "dependencies": []},
            {"id": "2", "description": "task2", "files": ["b.py"], "dependencies": ["1"]},
        ],
        "rationale": "two tasks",
    }
    p = Plan(**data)
    assert len(p.tasks) == 2
    assert p.tasks[0].id == "1"
    assert p.tasks[1].dependencies == ["1"]


# --- TestResult ---


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_defaults():
    t = TestResult(passed=True, summary="all good")
    assert t.failures == []


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="3 failures", failures=["test_a", "test_b", "test_c"])
    assert t.passed is False
    assert len(t.failures) == 3


# --- Settings ---


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


def test_settings_override_provider():
    s = Settings(provider=Provider.OPENAI, openai_api_key="sk-test")
    assert s.provider == Provider.OPENAI
    assert s.openai_api_key == "sk-test"


def test_settings_override_ollama():
    s = Settings(
        provider=Provider.OLLAMA,
        ollama_base_url="http://localhost:8080",
    )
    assert s.provider == Provider.OLLAMA
    assert s.ollama_base_url == "http://localhost:8080"
