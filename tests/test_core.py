import json

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_plan_round_trip():
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )
    raw = plan.model_dump()
    restored = Plan(**raw)
    assert restored.tasks[0].id == "1"
    assert restored.tasks[0].description == "do thing"
    assert restored.rationale == "ok"


def test_task_model_defaults():
    task = TaskModel(id="t1", description="desc")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_test_result_defaults():
    result = TestResult(passed=True, summary="ok")
    assert result.passed is True
    assert result.summary == "ok"
    assert result.failures == []


def test_provider_enum_values():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"
    assert list(Provider) == [Provider.ANTHROPIC, Provider.OPENAI, Provider.OLLAMA]


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.max_cycles == 0
    assert settings.max_parallel_tasks == 5
