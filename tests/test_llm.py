import json
from unittest.mock import AsyncMock, patch

import pytest
from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_llm_client_init():
    client = LLMClient()
    assert client.settings is not None


def test_llm_client_custom_settings():
    settings = Settings(model="test-model", provider=Provider.OPENAI)
    client = LLMClient(settings=settings)
    assert client.settings.model == "test-model"


@pytest.mark.asyncio
async def test_complete_openai_mocked():
    settings = Settings(provider=Provider.OPENAI, model="gpt-4")
    client = LLMClient(settings=settings)
    mock_response = AsyncMock()
    mock_response.choices[0].message.content = "Hello"
    with patch.object(client.openai.chat.completions, "create", return_value=mock_response):
        result = await client.complete("Say hi")
    assert result == "Hello"


@pytest.mark.asyncio
async def test_complete_anthropic_mocked():
    settings = Settings(provider=Provider.ANTHROPIC, model="claude-3")
    client = LLMClient(settings=settings)
    mock_response = AsyncMock()
    mock_response.content[0].text = "Hello"
    with patch.object(client.anthropic.messages, "create", return_value=mock_response):
        result = await client.complete("Say hi")
    assert result == "Hello"


@pytest.mark.asyncio
async def test_complete_ollama_mocked():
    settings = Settings(
        provider=Provider.OLLAMA,
        model="llama3",
        ollama_base_url="http://localhost:11434",
    )
    client = LLMClient(settings=settings)
    mock_response = {"response": "Hello from Ollama"}
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value.__aenter__.return_value.json.return_value = mock_response
        mock_client.post.return_value.__aenter__.return_value.raise_for_status = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        result = await client.complete("Say hi")
    assert result == "Hello from Ollama"


def test_unsupported_provider():
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    # Should not raise at init, only at complete
    assert client.settings.provider == Provider.OLLAMA
