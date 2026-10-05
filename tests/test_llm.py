import asyncio
from enum import Enum

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5
    assert s.ollama_base_url == "http://localhost:11434"


def test_provider_enum_has_ollama():
    assert Provider.OLLAMA.value == "ollama"


def test_llm_client_rejects_missing_anthropic_key():
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    client = LLMClient(settings=s)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = client.anthropic


def test_llm_client_rejects_missing_openai_key():
    s = Settings(provider=Provider.OPENAI, openai_api_key=None)
    client = LLMClient(settings=s)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = client.openai


def test_llm_client_unsupported_provider(monkeypatch):
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="k")
    client = LLMClient(settings=s)

    class Unsupported(Enum):
        WEIRD = "weird"

    # provider is typed as a Provider enum member; complete() logs
    # provider.value, so a bare string would fail with AttributeError instead.
    monkeypatch.setattr(client.settings, "provider", Unsupported.WEIRD)

    with pytest.raises(ValueError, match="Unsupported provider"):
        asyncio.run(client.complete("hi"))


def test_llm_client_list_files_missing_dir():
    s = Settings()
    client = LLMClient(settings=s)
    assert client.list_files("/nonexistent-dir-xyz") == []


def test_llm_client_list_files(tmp_path):
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y = 2")
    s = Settings()
    client = LLMClient(settings=s)
    files = sorted(client.list_files(tmp_path))
    assert files == ["a.py", "sub/b.py"]
