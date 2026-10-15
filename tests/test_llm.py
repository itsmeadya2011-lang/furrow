"""Tests for the LLMClient: provider routing and Ollama support."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def make_settings(**overrides):
    defaults = {
        "provider": Provider.ANTHROPIC,
        "model": "test-model",
        "anthropic_api_key": "test-key",
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_routes_to_anthropic():
    """When provider is ANTHROPIC, complete() calls _complete_anthropic."""
    settings = make_settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)

    with patch.object(client, "_complete_anthropic", new=AsyncMock(return_value="anthropic ok")):
        result = await client.complete("hello")
        assert result == "anthropic ok"


@pytest.mark.asyncio
async def test_complete_routes_to_openai():
    """When provider is OPENAI, complete() calls _complete_openai."""
    settings = make_settings(
        provider=Provider.OPENAI, openai_api_key="test-key"
    )
    client = LLMClient(settings=settings)

    with patch.object(client, "_complete_openai", new=AsyncMock(return_value="openai ok")):
        result = await client.complete("hello")
        assert result == "openai ok"


@pytest.mark.asyncio
async def test_complete_routes_to_ollama():
    """When provider is OLLAMA, complete() calls _complete_ollama."""
    settings = make_settings(provider=Provider.OLLAMA, model="llama2")
    client = LLMClient(settings=settings)

    with patch.object(client, "_complete_ollama", new=AsyncMock(return_value="ollama ok")):
        result = await client.complete("hello")
        assert result == "ollama ok"


@pytest.mark.asyncio
async def test_complete_unsupported_provider():
    """An unknown provider raises ValueError."""
    settings = make_settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)

    # Mock settings to return an unsupported provider string
    with patch.object(client, "settings", MagicMock(provider="unknown_provider")):
        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("hello")


@pytest.mark.asyncio
async def test_complete_uses_default_model():
    """When no model is passed, the default model from settings is used."""
    settings = make_settings(provider=Provider.ANTHROPIC, model="my-default-model")
    client = LLMClient(settings=settings)

    with patch.object(client, "_complete_anthropic", new=AsyncMock(return_value="ok")) as mock_method:
        await client.complete("hello")
        # Check that model was passed as "my-default-model"
        call_kwargs = mock_method.call_args
        assert call_kwargs.kwargs.get("model") == "my-default-model" or "my-default-model" in str(mock_method.call_args)


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def test_ollama_client_initializes_with_correct_base_url():
    """The ollama property creates an AsyncOpenAI with the configured base_url."""
    settings = make_settings(
        provider=Provider.OLLAMA,
        model="llama2",
        ollama_base_url="http://localhost:11434",
    )
    client = LLMClient(settings=settings)

    with patch("furrow.llm.AsyncOpenAI") as mock_openai:
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance
        ollama_client = client.ollama

        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args
        assert "11434" in str(call_kwargs.kwargs.get("base_url", ""))
        assert call_kwargs.kwargs.get("api_key") == "ollama"


@pytest.mark.asyncio
async def test_ollama_complete_returns_response():
    """_complete_ollama returns the content from the response."""
    settings = make_settings(
        provider=Provider.OLLAMA,
        model="llama2",
        ollama_base_url="http://localhost:11434",
    )
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Ollama says hello"
    mock_response.usage = None

    client._ollama = MagicMock()
    client._ollama.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await client._complete_ollama("test prompt", "system prompt", "llama2")
    assert result == "Ollama says hello"
    client._ollama.chat.completions.create.assert_awaited_once()


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_complete_returns_response():
    """_complete_anthropic returns the content from the response."""
    settings = make_settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Anthropic says hello")]
    mock_response.usage = MagicMock(output_tokens=100)

    client._anthropic = MagicMock()
    client._anthropic.messages.create = AsyncMock(return_value=mock_response)

    result = await client._complete_anthropic("test prompt", "system prompt", "claude")
    assert result == "Anthropic says hello"
    client._anthropic.messages.create.assert_awaited_once()


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openai_complete_returns_response():
    """_complete_openai returns the content from the response."""
    settings = make_settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "OpenAI says hello"
    mock_response.usage = MagicMock(completion_tokens=100)

    client._openai = MagicMock()
    client._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await client._complete_openai("test prompt", "system prompt", "gpt-4")
    assert result == "OpenAI says hello"
    client._openai.chat.completions.create.assert_awaited_once()
