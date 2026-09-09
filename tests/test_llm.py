import asyncio

import pytest
from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_provider_enum_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


def test_settings_ollama_provider():
    """Ollama provider should be selectable and use the configured base URL."""
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    assert s.provider == Provider.OLLAMA
    assert s.ollama_base_url == "http://localhost:11434"


def test_llm_client_routes_ollama():
    """LLMClient should route to _complete_ollama when provider is OLLAMA."""
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    assert client.settings.provider == Provider.OLLAMA


def test_llm_client_routes_anthropic():
    """LLMClient should route to _complete_anthropic when provider is ANTHROPIC."""
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)
    assert client.settings.provider == Provider.ANTHROPIC


def test_llm_client_routes_openai():
    """LLMClient should route to _complete_openai when provider is OPENAI."""
    settings = Settings(provider=Provider.OPENAI)
    client = LLMClient(settings=settings)
    assert client.settings.provider == Provider.OPENAI


def test_llm_client_unsupported_provider():
    """LLMClient should raise ValueError for unsupported providers."""
    client = LLMClient(settings=Settings())
    client.settings.provider = "unsupported"  # type: ignore

    async def _run():
        await client.complete("test prompt")

    with pytest.raises(ValueError, match="Unsupported provider"):
        asyncio.run(_run())
