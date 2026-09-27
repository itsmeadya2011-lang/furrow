import os

import pytest

from furrow.config import Provider, Settings


@pytest.fixture(autouse=True)
def _clear_furrow_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("FURROW_"):
            monkeypatch.delenv(key)


def test_settings_creates_with_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.model is not None
    assert s.ollama_base_url == "http://localhost:11434"


def test_max_tool_calls_default():
    assert Settings().max_tool_calls == 30


def test_max_parallel_tasks_default():
    assert Settings().max_parallel_tasks == 5


def test_max_cycles_default():
    assert Settings().max_cycles == 0


def test_provider_enum_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"
    assert {p.value for p in Provider} == {"anthropic", "openai", "ollama"}
