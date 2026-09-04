import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.planner_model == "claude-3-5-haiku-20241022"
    assert s.worker_model == "claude-3-5-sonnet-20241022"
    assert s.tester_model == "claude-3-5-sonnet-20241022"
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.request_timeout == 120
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.id == "1"
    assert t.description == "do thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_creation():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert len(p.tasks) == 1
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
    assert t.summary == "ok"
    assert t.failures == []
