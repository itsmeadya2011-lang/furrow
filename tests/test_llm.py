"""Unit tests for the LLMClient, focusing on provider routing and Ollama support."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_settings_ollama_config():
    """Ollama settings should configure the base URL."""
    s = Settings(
        provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434"
    )
    client = LLMClient(settings=s)
    assert client.settings.provider == Provider.OLLAMA
    assert client.settings.ollama_base_url == "http://localhost:11434"


@pytest.mark.asyncio
async def test_ollama_complete():
    """The Ollama provider should POST to the generate endpoint and return the response."""
    s = Settings(
        provider=Provider.OLLAMA,
        ollama_base_url="http://localhost:11434",
        model="llama2",
    )
    client = LLMClient(settings=s)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "Ollama response!"}

    mock_http_instance = MagicMock()
    mock_http_instance.post = AsyncMock(return_value=mock_response)
    mock_http_instance.__aenter__ = AsyncMock(return_value=mock_http_instance)
    mock_http_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_http_instance):
        result = await client.complete("Hello, Ollama!", model="llama2")

    assert result == "Ollama response!"
    mock_http_instance.post.assert_called_once()


@pytest.mark.asyncio
async def test_ollama_strips_trailing_slash():
    """Ollama base URL with trailing slash should still work."""
    s = Settings(
        provider=Provider.OLLAMA,
        ollama_base_url="http://localhost:11434/",
        model="llama2",
    )
    client = LLMClient(settings=s)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "ok"}

    mock_http_instance = MagicMock()
    mock_http_instance.post = AsyncMock(return_value=mock_response)
    mock_http_instance.__aenter__ = AsyncMock(return_value=mock_http_instance)
    mock_http_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_http_instance):
        result = await client.complete("test", model="llama2")

    assert result == "ok"


@pytest.mark.asyncio
async def test_ollama_url_construction():
    """The URL sent to Ollama should use the configured base URL."""
    s = Settings(
        provider=Provider.OLLAMA,
        ollama_base_url="http://localhost:11434",
        model="llama2",
    )
    client = LLMClient(settings=s)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "ok"}

    mock_http_instance = MagicMock()
    mock_http_instance.post = AsyncMock(return_value=mock_response)
    mock_http_instance.__aenter__ = AsyncMock(return_value=mock_http_instance)
    mock_http_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_http_instance):
        await client.complete("test", model="llama2")

    call_args = mock_http_instance.post.call_args
    url = call_args[0][0] if call_args[0] else call_args[1].get("url")
    assert "http://localhost:11434/api/generate" in str(url)
