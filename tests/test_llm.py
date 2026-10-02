"""Tests for the LLMClient."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_llm_client_init_default():
    """LLMClient initializes with default settings."""
    client = LLMClient()
    assert client.settings.provider == Provider.ANTHROPIC


def test_llm_client_init_with_settings():
    """LLMClient accepts custom settings."""
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)
    assert client.settings.provider == Provider.OPENAI


def test_llm_client_ollama_uses_openai_compatible():
    """Ollama provider uses OpenAI-compatible client with ollama base URL."""
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)
    # The ollama property should use the OpenAI client with the ollama base URL
    _ = client.ollama
    assert client._openai is not None


async def test_llm_client_complete_anthropic():
    """LLMClient routes to Anthropic for anthropic provider."""
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello from Anthropic")]

    with patch.object(LLMClient, "anthropic", new_callable=PropertyMock) as mock_anthropic:
        mock_anthropic.return_value.messages.create = AsyncMock(return_value=mock_response)
        result = await client.complete("Hello", model="claude-3-5-sonnet")
    assert result == "Hello from Anthropic"


async def test_llm_client_complete_openai():
    """LLMClient routes to OpenAI for openai provider."""
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test")
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenAI"))]

    with patch.object(LLMClient, "openai", new_callable=PropertyMock) as mock_openai:
        mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await client.complete("Hello", model="gpt-4")
    assert result == "Hello from OpenAI"


async def test_llm_client_complete_ollama():
    """LLMClient routes to Ollama (via OpenAI-compatible API) for ollama provider."""
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello from Ollama"))]

    with patch.object(LLMClient, "ollama", new_callable=PropertyMock) as mock_ollama:
        mock_ollama.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await client.complete("Hello", model="llama3")
    assert result == "Hello from Ollama"


async def test_llm_client_complete_unsupported():
    """LLMClient raises ValueError for unsupported provider."""
    # Provider.ANTHROPIC but with a fake enum value
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test"))

    # Force an unsupported provider by monkey-patching
    client.settings.provider = "unsupported"
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("Hello")


async def test_llm_client_complete_with_default_model():
    """LLMClient uses default model when model is None."""
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Response")]

    with patch.object(LLMClient, "anthropic", new_callable=PropertyMock) as mock_anthropic:
        mock_anthropic.return_value.messages.create = AsyncMock(return_value=mock_response)
        await client.complete("Hello")
        call_kwargs = mock_anthropic.return_value.messages.create.call_args.kwargs
        assert call_kwargs["model"] == settings.model


async def test_llm_client_read_file():
    """LLMClient can read files."""
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test"))
    content = await client.read_file("furrow/config.py")
    assert "class Settings" in content


def test_llm_client_list_files():
    """LLMClient can list files in a directory."""
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test"))
    files = client.list_files("furrow/")
    assert len(files) > 0
    assert any("config.py" in f for f in files)


def test_llm_client_anthropic_api_key_missing():
    """LLMClient raises ValueError when Anthropic API key is missing."""
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    with patch.dict("os.environ", {}, clear=False), pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        _ = client.anthropic


def test_llm_client_openai_api_key_missing():
    """LLMClient raises ValueError when OpenAI API key is missing."""
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    with patch.dict("os.environ", {}, clear=False), pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        _ = client.openai
