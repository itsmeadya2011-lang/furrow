from furrow.config import Provider, Settings, TaskModel, TestResult


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "10")
    s = Settings()
    assert s.provider == Provider.OPENAI
    assert s.max_parallel_tasks == 10


def test_provider_enum_values():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_task_model_status_default():
    t = TaskModel(id="1", description="x")
    assert t.status == "pending"


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="broken", failures=["a", "b"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_settings_ollama_default_url():
    s = Settings()
    assert s.ollama_base_url == "http://localhost:11434"


def test_settings_workspace_default():
    from pathlib import Path
    s = Settings()
    assert s.workspace == Path.cwd()
