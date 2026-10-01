from __future__ import annotations

import pytest

from furrow.config import Provider, Settings


@pytest.fixture
def clean_env(monkeypatch):
    for var in [
        "FURROW_PROVIDER",
        "FURROW_OLLAMA_MODEL",
        "FURROW_TASK_TIMEOUT",
        "FURROW_MAX_CYCLES",
        "FURROW_MAX_PARALLEL_TASKS",
        "FURROW_MODEL",
        "FURROW_WORKER_MODEL",
    ]:
        monkeypatch.delenv(var, raising=False)


def test_default_settings(clean_env):
    s = Settings()
    assert s.task_timeout == 300
    assert s.ollama_model == "llama3.1"
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5
    assert s.provider == Provider.ANTHROPIC


def test_ollama_model_env_var(clean_env, monkeypatch):
    monkeypatch.setenv("FURROW_OLLAMA_MODEL", "custom-ollama")
    s = Settings()
    assert s.ollama_model == "custom-ollama"


def test_task_timeout_env_var(clean_env, monkeypatch):
    monkeypatch.setenv("FURROW_TASK_TIMEOUT", "600")
    s = Settings()
    assert s.task_timeout == 600


def test_provider_ollama(clean_env, monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "ollama")
    s = Settings()
    assert s.provider == Provider.OLLAMA
