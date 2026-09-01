from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


async def test_complete_dispatches_to_ollama(monkeypatch):
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    monkeypatch.setattr(LLMClient, "_complete_ollama", AsyncMock(return_value="ollama response"))
    result = await client.complete("hi")
    assert result == "ollama response"


async def test_complete_raises_for_unsupported_provider(monkeypatch):
    class FakeProvider:
        pass

    settings = Settings()
    object.__setattr__(settings, "provider", FakeProvider())
    client = LLMClient(settings=settings)
    monkeypatch.setattr(LLMClient, "_complete_ollama", AsyncMock())
    with pytest.raises(ValueError) as excinfo:
        await client.complete("hi")
    assert "Unsupported provider" in str(excinfo.value)


async def test_anthropic_retry_succeeds_after_failures(monkeypatch):
    import tenacity

    settings = Settings(provider=Provider.ANTHROPIC)
    settings.anthropic_api_key = "test-key"
    client = LLMClient(settings=settings)

    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="hello")]

    call_count = {"n": 0}

    async def fake_create(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise RuntimeError("transient error")
        return fake_response

    fake_messages = MagicMock()
    fake_messages.create = fake_create
    fake_anthropic = MagicMock()
    fake_anthropic.messages = fake_messages
    client._anthropic = fake_anthropic

    monkeypatch.setattr(tenacity.nap, "sleep", lambda _seconds: None)
    result = await client._complete_anthropic("hi", "", "claude-test")
    assert result == "hello"
    assert call_count["n"] == 3


def test_anthropic_complete_is_wrapped_by_tenacity():
    assert hasattr(LLMClient._complete_anthropic, "retry")


def test_default_settings_provider():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
