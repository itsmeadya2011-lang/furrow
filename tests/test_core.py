import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="example")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_settings_max_tool_calls():
    s = Settings()
    assert s.max_tool_calls > 0


def test_settings_ollama_provider():
    s = Settings(provider=Provider.OLLAMA)
    assert s.provider == Provider.OLLAMA
    assert s.ollama_base_url.startswith("http://")
