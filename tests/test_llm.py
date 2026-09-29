from furrow.config import Provider, Settings
from furrow.llm import LLMClient


async def test_llm_dispatch_anthropic():
    s = Settings()
    s.provider = Provider.ANTHROPIC
    client = LLMClient(settings=s)
    called = {}

    async def fake(prompt, system, model, max_tokens):
        called["provider"] = "anthropic"
        called["model"] = model
        called["max_tokens"] = max_tokens
        return "ok"

    client._complete_anthropic = fake
    result = await client.complete("hi", system="sys", model="m")
    assert result == "ok"
    assert called["provider"] == "anthropic"
    assert called["model"] == "m"
    assert called["max_tokens"] == s.max_tokens


async def test_llm_dispatch_openai():
    s = Settings()
    s.provider = Provider.OPENAI
    client = LLMClient(settings=s)
    called = {}

    async def fake(prompt, system, model, max_tokens):
        called["provider"] = "openai"
        return "ok"

    client._complete_openai = fake
    result = await client.complete("hi", model="m")
    assert result == "ok"
    assert called["provider"] == "openai"


async def test_llm_dispatch_ollama():
    """Ollama must be a supported provider (previously raised ValueError)."""
    s = Settings()
    s.provider = Provider.OLLAMA
    client = LLMClient(settings=s)
    called = {}

    async def fake(prompt, system, model):
        called["provider"] = "ollama"
        called["model"] = model
        return "ok"

    client._complete_ollama = fake
    result = await client.complete("hi", system="sys", model="m")
    assert result == "ok"
    assert called["provider"] == "ollama"
    assert called["model"] == "m"


def test_llm_retryable_errors_returns_tuple():
    errs = LLMClient.retryable_errors()
    assert isinstance(errs, tuple)
    assert len(errs) >= 1


def test_llm_client_accepts_none_settings():
    # Should fall back to the global settings when none provided.
    client = LLMClient(settings=None)
    assert client.settings is not None
