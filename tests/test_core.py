import json
import tempfile
from pathlib import Path

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="test")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_files():
    task = TaskModel(id="1", description="test", files=["a.py", "b.py"])
    assert task.files == ["a.py", "b.py"]


def test_plan_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2", dependencies=["1"]),
    ]
    p = Plan(tasks=tasks, rationale="two tasks")
    assert len(p.tasks) == 2
    assert p.tasks[1].dependencies == ["1"]


def test_settings_defaults():
    s = Settings()
    assert s.provider.value == "anthropic"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_workspace_expansion():
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(workspace=tmpdir)
        assert s.workspace == Path(tmpdir).resolve()


def test_settings_max_parallel_validation():
    with pytest.raises(Exception):
        Settings(max_parallel_tasks=0)
    with pytest.raises(Exception):
        Settings(max_parallel_tasks=21)


def test_settings_log_level_validation():
    with pytest.raises(Exception):
        Settings(log_level="INVALID")


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="2 failed", failures=["test_a", "test_b"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_task_status_transitions():
    task = TaskModel(id="1", description="test")
    assert task.status == "pending"
    task.status = "completed"
    assert task.status == "completed"
    task.status = "failed"
    assert task.status == "failed"


def test_plan_serialization():
    p = Plan(
        tasks=[TaskModel(id="1", description="test", files=["a.py"])],
        rationale="test",
    )
    data = p.model_dump()
    assert data["tasks"][0]["files"] == ["a.py"]
    assert data["rationale"] == "test"

    # Round-trip
    p2 = Plan(**data)
    assert p2.tasks[0].files == ["a.py"]


def test_test_result_serialization():
    t = TestResult(passed=True, summary="all good", failures=[])
    data = t.model_dump()
    assert data["passed"] is True

    t2 = TestResult(**data)
    assert t2.passed is True


def test_settings_extra_fields_ignored():
    # Should not raise with extra fields due to extra="ignore"
    s = Settings(extra_field="ignored")
    assert s.provider.value == "anthropic"
