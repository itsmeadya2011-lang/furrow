from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from openai import AsyncOpenAI

from furrow.config import Provider, Settings
from furrow.llm import LLMClient, _sanitize


def test_sanitize_api_key():
    text = "Here is a key: sk-1234567890abcdefghijklmnop"
    result = _sanitize(text)
    assert "sk-1234567890abcdefghijklmnop" not in result
    assert "[REDACTED]" in result


def test_sanitize_no_key():
    text = "Hello, world! This is a normal message."
    result = _sanitize(text)
    assert result == text


@pytest.mark.asyncio
async def test_ollama_provider_dispatch():
    s = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=s)
    with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock_ollama:
        await client.complete("prompt", model="some-model")
        mock_ollama.assert_awaited_once_with("prompt", "", "some-model")


@pytest.mark.asyncio
async def test_ollama_model_default():
    s = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=s)
    with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock_ollama:
        await client.complete("prompt")
        mock_ollama.assert_awaited_once_with("prompt", "", s.ollama_model)


@pytest.mark.asyncio
async def test_ollama_property():
    client = LLMClient()
    result = client.ollama
    assert isinstance(result, AsyncOpenAI)
    assert result.api_key == "ollama"
    assert result is client.ollama
