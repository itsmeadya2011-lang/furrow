import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_llmclient_validate_anthropic_missing_key():
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]
    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        client.validate()


def test_llmclient_validate_openai_missing_key():
    settings = Settings(provider=Provider.OPENAI, openai_api_key=None)
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        client.validate()


def test_llmclient_validate_ollama_invalid_url():
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="ftp://bad-url")
    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    with pytest.raises(ValueError, match="OLLAMA_BASE_URL must be a valid URL"):
        client.validate()


def test_llmclient_validate_ollama_valid_url():
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    client.validate()


def test_llmclient_validate_anthropic_with_key():
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    client.validate()


def test_llmclient_validate_openai_with_key():
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    client.validate()


@pytest.mark.asyncio
async def test_complete_ollama_with_mocked_client():
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    client._anthropic = None
    client._openai = None

    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "Hello from Ollama"}}
    mock_response.raise_for_status = MagicMock()
    mock_post = AsyncMock(return_value=mock_response)
    mock_client = MagicMock()
    mock_client.post = mock_post

    async_context = MagicMock()
    async_context.__aenter__ = AsyncMock(return_value=mock_client)
    async_context.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=async_context):
        result = await client._complete_ollama("hello", "", "llama2")

    assert result == "Hello from Ollama"
    expected_url = "http://localhost:11434/api/chat"
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == expected_url
    assert call_args[1]["json"]["model"] == "llama2"


@pytest.mark.asyncio
async def test_complete_dispatches_anthropic():
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    client._anthropic = AsyncMock()
    client._openai = None
    client._anthropic.messages.create = AsyncMock(
        return_value=MagicMock(content=[MagicMock(text="anthropic response")])
    )

    with patch.object(client, "_complete_anthropic", new_callable=AsyncMock, return_value="anthropic response") as mock_method:
        await client.complete("hello")

    mock_method.assert_called_once_with("hello", "", client.settings.model)


@pytest.mark.asyncio
async def test_complete_dispatches_openai():
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    client._anthropic = None
    client._openai = AsyncMock()

    with patch.object(client, "_complete_openai", new_callable=AsyncMock, return_value="openai response") as mock_method:
        await client.complete("hello")

    mock_method.assert_called_once_with("hello", "", client.settings.model)


@pytest.mark.asyncio
async def test_complete_dispatches_ollama():
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    client._anthropic = None
    client._openai = None

    with patch.object(client, "_complete_ollama", new_callable=AsyncMock, return_value="ollama response") as mock_method:
        await client.complete("hello")

    mock_method.assert_called_once_with("hello", "", client.settings.model)