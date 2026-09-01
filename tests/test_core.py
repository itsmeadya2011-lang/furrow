import pytest
from furrow.config import Plan, Settings, TaskModel, TestResult


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.id == "1"
    assert t.description == "do thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_values():
    t = TaskModel(id="2", description="refactor", files=["a.py"], dependencies=["1"], status="completed", result="done")
    assert t.id == "2"
    assert t.files == ["a.py"]
    assert t.dependencies == ["1"]
    assert t.status == "completed"
    assert t.result == "done"


def test_plan_validation():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"


def test_plan_missing_rationale():
    with pytest.raises(Exception):
        Plan(tasks=[TaskModel(id="1", description="do thing")])


def test_test_result_defaults():
    t = TestResult(passed=True, summary="ok")
    assert t.passed is True
    assert t.summary == "ok"
    assert t.failures == []


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="bad", failures=["error1", "error2"])
    assert t.passed is False
    assert t.failures == ["error1", "error2"]


def test_settings_defaults():
    s = Settings()
    assert s.provider.value == "anthropic"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.ollama_base_url == "http://localhost:11434"
