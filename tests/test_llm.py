from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def _client(**overrides):
    return LLMClient(settings=Settings(**overrides))


def test_ollama_property():
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://custom-ollama:9999")
    client = LLMClient(settings=settings)
    ollama_client = client.ollama
    assert isinstance(ollama_client, AsyncOpenAI)
    assert str(ollama_client.base_url) == settings.ollama_base_url


async def test_ollama_complete_branch():
    client = _client(provider=Provider.OLLAMA)
    client._complete_ollama = AsyncMock(return_value="ollama result")
    result = await client.complete("hello")
    client._complete_ollama.assert_awaited_once()
    assert result == "ollama result"


async def test_ollama_complete_returns_content():
    client = _client(provider=Provider.OLLAMA)
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = "response text"
    response.choices = [choice]
    client._ollama = MagicMock()
    client._ollama.chat.completions.create = AsyncMock(return_value=response)
    result = await client.complete("hello")
    assert result == "response text"


async def test_complete_anthropic():
    client = _client(provider=Provider.ANTHROPIC)
    response = MagicMock()
    content_block = MagicMock()
    content_block.text = "anthropic response"
    response.content = [content_block]
    client._anthropic = MagicMock()
    client._anthropic.messages.create = AsyncMock(return_value=response)
    result = await client.complete("hello")
    assert result == "anthropic response"


async def test_complete_openai():
    client = _client(provider=Provider.OPENAI)
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = "openai response"
    response.choices = [choice]
    client._openai = MagicMock()
    client._openai.chat.completions.create = AsyncMock(return_value=response)
    result = await client.complete("hello")
    assert result == "openai response"


async def test_complete_unsupported():
    client = _client()
    client.settings = MagicMock()
    client.settings.provider = "unknown-provider"
    client.settings.model = "test-model"
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hello")
