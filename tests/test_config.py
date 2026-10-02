from __future__ import annotations

from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_provider_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.status == "pending"
    assert t.result is None
    assert t.files == []
    assert t.dependencies == []


def test_plan_requires_tasks_and_rationale():
    p = Plan(tasks=[TaskModel(id="1", description="x")], rationale="r")
    assert p.rationale == "r"
    assert len(p.tasks) == 1


def test_plan_requires_tasks():
    try:
        Plan(rationale="r")
    except ValidationError:
        pass
    else:
        raise AssertionError("Plan should require tasks")


def test_test_result_defaults():
    t = TestResult(passed=True, summary="ok")
    assert t.passed is True
    assert t.failures == []


def test_settings_defaults():
    s = Settings(provider=Provider.OLLAMA)
    assert s.max_retries == 3
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5
    assert s.ollama_base_url == "http://localhost:11434"


def test_settings_default_provider_is_anthropic():
    assert Settings().provider == Provider.ANTHROPIC
