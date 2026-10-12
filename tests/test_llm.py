import asyncio

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


async def test_complete_retries_then_succeeds(monkeypatch):
    async def _instant_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    settings = Settings(max_retries=3)
    client = LLMClient(settings=settings)

    calls = {"count": 0}

    async def failing_twice_then_ok(prompt, system, model):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise asyncio.TimeoutError("timeout")
        return "done"

    monkeypatch.setattr(client, "_dispatch", failing_twice_then_ok)

    result = await client.complete("hello")
    assert result == "done"
    assert calls["count"] == 3


async def test_complete_non_retryable_propagates(monkeypatch):
    settings = Settings(max_retries=3)
    client = LLMClient(settings=settings)

    calls = {"count": 0}

    async def boom(prompt, system, model):
        calls["count"] += 1
        raise ValueError("boom")

    monkeypatch.setattr(client, "_dispatch", boom)

    with pytest.raises(ValueError, match="boom"):
        await client.complete("hello")
    assert calls["count"] == 1


def test_ollama_client_base_url():
    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    )
    assert str(client.ollama.base_url).startswith("http://localhost:11434/v1")


def test_anthropic_property_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(settings=Settings(anthropic_api_key=None))
    with pytest.raises(ValueError):
        _ = client.anthropic