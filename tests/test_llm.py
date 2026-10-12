"""Tests for LLM client improvements: OLLAMA provider, max_tokens, retry."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def make_settings(provider=Provider.ANTHROPIC, **overrides) -> Settings:
    return Settings(provider=provider, **overrides)


@pytest.mark.asyncio
async def test_complete_dispatches_ollama():
    """complete() should dispatch to _complete_ollama when provider is OLLAMA."""
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)

    with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock_ollama:
        mock_ollama.return_value = "ollama response"
        result = await client.complete("hello")
        assert result == "ollama response"
        mock_ollama.assert_called_once()


@pytest.mark.asyncio
async def test_complete_dispatches_anthropic():
    """complete() should dispatch to _complete_anthropic when provider is ANTHROPIC."""
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient(settings=s)

    with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock_anthropic:
        mock_anthropic.return_value = "anthropic response"
        result = await client.complete("hello")
        assert result == "anthropic response"
        mock_anthropic.assert_called_once()


@pytest.mark.asyncio
async def test_complete_uses_max_tokens_from_settings():
    """_complete_anthropic should use max_tokens from settings, not a hardcoded value."""
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key", max_tokens=2048)
    client = LLMClient(settings=s)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]

    with patch.object(client, "anthropic") as mock_anthropic_client:
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)
        await client._complete_anthropic("prompt", "system", "model")

        call_kwargs = mock_anthropic_client.messages.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_ollama_uses_base_url_from_settings():
    """_complete_ollama should use the ollama_base_url from settings."""
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://custom:8000")
    client = LLMClient(settings=s)

    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"response": "ollama result"})
    mock_response.raise_for_status = MagicMock()

    mock_http_client = MagicMock()
    mock_http_client.post = AsyncMock(return_value=mock_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=None)

    with patch("furrow.llm.httpx.AsyncClient", return_value=mock_http_client):
        result = await client._complete_ollama("prompt", "system", "model")

        assert result == "ollama result"
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert call_args.args[0] == "http://custom:8000/api/generate"
        assert call_args.kwargs["json"]["model"] == "model"
        assert call_args.kwargs["json"]["prompt"] == "prompt"
