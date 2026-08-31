from pathlib import Path

from furrow.config import Provider, Settings, configure_logging


def test_configure_logging_does_not_raise():
    configure_logging("INFO")
    configure_logging("DEBUG")


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"
    assert s.ollama_base_url == "http://localhost:11434"
    assert isinstance(s.workspace, Path)
    assert s.planner_model != ""
    assert s.worker_model != ""
    assert s.tester_model != ""


def test_settings_overrides(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "12")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    monkeypatch.setenv("FURROW_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("FURROW_MODEL", "custom-model")

    s = Settings()
    assert s.provider == Provider.OPENAI
    assert s.max_parallel_tasks == 12
    assert s.max_cycles == 7
    assert s.log_level == "DEBUG"
    assert s.model == "custom-model"


def test_settings_construct_with_kwargs():
    s = Settings(provider=Provider.OLLAMA, max_parallel_tasks=2, max_cycles=3)
    assert s.provider == Provider.OLLAMA
    assert s.max_parallel_tasks == 2
    assert s.max_cycles == 3