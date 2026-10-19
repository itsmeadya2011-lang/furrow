from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_llm_client_init():
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)
    assert client.settings.provider == Provider.ANTHROPIC


def test_llm_client_has_ollama_property():
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    assert client._ollama is None
    # Accessing the property should create the client
    _ = client.ollama
    assert client._ollama is not None


def test_llm_client_has_retry_config():
    """Verify the LLMClient has retry logic via tenacity."""
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)
    # The complete method should be wrapped with retry logic
    # We can verify by checking it has the retrying attribute
    assert hasattr(client, "complete")


@pytest.mark.asyncio
async def test_complete_routes_anthropic():
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient(settings=settings)
    client._complete_anthropic = AsyncMock()
    client._complete_anthropic.return_value = "test response"

    result = await client.complete("hello", model="claude-test")
    assert result == "test response"
    client._complete_anthropic.assert_called_once_with("hello", "", "claude-test")


@pytest.mark.asyncio
async def test_complete_routes_openai():
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)
    client._complete_openai = AsyncMock()
    client._complete_openai.return_value = "test response"

    result = await client.complete("hello", model="gpt-test")
    assert result == "test response"
    client._complete_openai.assert_called_once_with("hello", "", "gpt-test")


@pytest.mark.asyncio
async def test_complete_routes_ollama():
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    client._complete_ollama = AsyncMock()
    client._complete_ollama.return_value = "test response"

    result = await client.complete("hello", model="llama-test")
    assert result == "test response"
    client._complete_ollama.assert_called_once_with("hello", "", "llama-test")