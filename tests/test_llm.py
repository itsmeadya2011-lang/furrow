"""Tests for the LLMClient wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def _settings(**overrides):
    """Build a Settings instance with optional overrides."""
    return Settings(**overrides)


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_unsupported_provider_raises():
    """An unknown provider should raise ValueError."""
    s = _settings(provider=Provider.ANTHROPIC, anthropic_api_key="x")
    client = LLMClient(settings=s)
    client.settings.provider = "invalid"
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hello")


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_anthropic():
    s = _settings(provider=Provider.ANTHROPIC, anthropic_api_key="x")
    client = LLMClient(settings=s)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello from Anthropic!")]

    # Bypass the lazy property by pre-setting the internal client.
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
    client._anthropic = mock_anthropic

    result = await client.complete("test prompt", system="sys")
    assert result == "Hello from Anthropic!"
    call_kwargs = mock_anthropic.messages.create.call_args.kwargs
    assert call_kwargs["model"] == s.model
    assert call_kwargs["max_tokens"] == s.max_tokens  # default from settings


@pytest.mark.asyncio
async def test_complete_anthropic_custom_max_tokens():
    s = _settings(provider=Provider.ANTHROPIC, anthropic_api_key="x")
    client = LLMClient(settings=s)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]

    mock_anthropic = MagicMock()
    mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
    client._anthropic = mock_anthropic

    await client.complete("test", max_tokens=2048)
    call_kwargs = mock_anthropic.messages.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_openai():
    s = _settings(provider=Provider.OPENAI, openai_api_key="x")
    client = LLMClient(settings=s)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenAI!"))]

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
    client._openai = mock_openai

    result = await client.complete("test prompt")
    assert result == "Hello from OpenAI!"


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_ollama():
    """Ollama provider should POST to /api/generate and parse the response."""
    s = _settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)

    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"response": "Hello from Ollama!"})
    mock_response.raise_for_status = MagicMock()

    mock_httpx = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)
    client._httpx = mock_httpx

    result = await client.complete("test prompt", model="llama3")
    assert result == "Hello from Ollama!"

    call_args = mock_httpx.post.call_args
    assert call_args.args[0] == "/api/generate"
    payload = call_args.kwargs["json"]
    assert payload["model"] == "llama3"
    assert payload["prompt"] == "test prompt"
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == s.max_tokens


@pytest.mark.asyncio
async def test_complete_ollama_default_model():
    s = _settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)

    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"response": "ok"})
    mock_response.raise_for_status = MagicMock()

    mock_httpx = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)
    client._httpx = mock_httpx

    await client.complete("test")
    payload = mock_httpx.post.call_args.kwargs["json"]
    assert payload["model"] == s.model


# ---------------------------------------------------------------------------
# max_tokens from settings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_tokens_defaults_to_settings():
    """When max_tokens is not passed, it should come from settings."""
    s = _settings(
        provider=Provider.ANTHROPIC, anthropic_api_key="x", max_tokens=8192
    )
    client = LLMClient(settings=s)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]

    mock_anthropic = MagicMock()
    mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
    client._anthropic = mock_anthropic

    await client.complete("test")
    call_kwargs = mock_anthropic.messages.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 8192


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["hello", ""])
async def test_read_and_write_file(tmp_path, content):
    s = _settings(provider=Provider.ANTHROPIC, anthropic_api_key="x")
    client = LLMClient(settings=s)
    target = tmp_path / "sub" / "file.txt"

    await client.write_file(target, content)
    assert await client.read_file(target) == content


def test_list_files(tmp_path):
    s = _settings(provider=Provider.ANTHROPIC, anthropic_api_key="x")
    client = LLMClient(settings=s)

    (tmp_path / "a.py").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y")

    files = client.list_files(tmp_path)
    assert "a.py" in files
    assert "sub/b.py" in files


def test_list_files_nonexistent():
    s = _settings(provider=Provider.ANTHROPIC, anthropic_api_key="x")
    client = LLMClient(settings=s)
    assert client.list_files("/nonexistent/path") == []
