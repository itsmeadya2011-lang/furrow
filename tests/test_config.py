"""Tests for furrow.config: Provider enum, models, and Settings env prefix."""

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_provider_enum_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_provider_enum_lookup_from_value():
    assert Provider("anthropic") is Provider.ANTHROPIC
    assert Provider("openai") is Provider.OPENAI
    assert Provider("ollama") is Provider.OLLAMA


def test_provider_is_str_enum():
    assert isinstance(Provider.ANTHROPIC, str)
    assert Provider.ANTHROPIC == "anthropic"


def test_task_model_default_values():
    task = TaskModel(id="1", description="do thing")
    assert task.status == "pending"
    assert task.result is None
    assert task.files == []
    assert task.dependencies == []


def test_task_model_with_explicit_values():
    task = TaskModel(
        id="t1",
        description="desc",
        files=["a.py"],
        dependencies=["t0"],
        status="completed",
        result="done",
    )
    assert task.id == "t1"
    assert task.files == ["a.py"]
    assert task.dependencies == ["t0"]
    assert task.status == "completed"
    assert task.result == "done"


def test_plan_creation_with_tasks():
    tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]
    plan = Plan(tasks=tasks, rationale="reasoning")
    assert len(plan.tasks) == 2
    assert plan.rationale == "reasoning"


def test_plan_empty_tasks():
    plan = Plan(tasks=[], rationale="none")
    assert plan.tasks == []
    assert plan.rationale == "none"


def test_test_result_creation():
    result = TestResult(passed=True, summary="all good", failures=[])
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


def test_test_result_failures_default():
    result = TestResult(passed=False, summary="bad")
    assert result.passed is False
    assert result.summary == "bad"
    assert result.failures == []


def test_test_result_with_failures():
    result = TestResult(passed=False, summary="broken", failures=["failure 1", "failure 2"])
    assert result.passed is False
    assert len(result.failures) == 2


def test_settings_env_prefix_provider(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MODEL", "gpt-4")
    settings = Settings()
    assert settings.provider == Provider.OPENAI
    assert settings.model == "gpt-4"


def test_settings_env_prefix_ollama(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "ollama")
    monkeypatch.setenv("FURROW_OLLAMA_BASE_URL", "http://localhost:8080")
    settings = Settings()
    assert settings.provider == Provider.OLLAMA
    assert settings.ollama_base_url == "http://localhost:8080"


def test_settings_env_prefix_overrides_constructor(monkeypatch):
    monkeypatch.setenv("FURROW_MODEL", "env-model")
    settings = Settings(model="ctor-model")
    assert settings.model == "ctor-model"


def test_settings_default_max_cycles():
    settings = Settings()
    assert settings.max_cycles == 0
