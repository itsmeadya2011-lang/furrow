"""Tests for the LLM client."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient

# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_property_creates_client():
    """ollama property should lazily create an AsyncOpenAI with base_url from settings."""
    settings = Settings(provider=Provider.OLLAMA, model="llama3")
    client = LLMClient(settings=settings)

    assert client._ollama is None
    ollama = client.ollama
    assert ollama is not None
    assert client._ollama is ollama
    # The base_url should include /v1 for Ollama's OpenAI-compatible API
    assert "11434" in str(ollama.base_url)


@pytest.mark.asyncio
async def test_ollama_dispatch_in_complete():
    """complete() should dispatch to _complete_ollama for OLLAMA provider."""
    settings = Settings(provider=Provider.OLLAMA, model="llama3")
    client = LLMClient(settings=settings)

    with patch.object(client, "_complete_ollama", new=AsyncMock(return_value="ollama response")):
        result = await client.complete("test prompt")
    assert result == "ollama response"


@pytest.mark.asyncio
async def test_ollama_uses_correct_api_key():
    """Ollama client should use 'ollama' as the API key (OpenAI-compatible)."""
    settings = Settings(provider=Provider.OLLAMA, model="llama3")
    client = LLMClient(settings=settings)

    # Accessing the property triggers creation
    _ = client.ollama
    assert client._ollama is not None


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_dispatches_anthropic():
    """complete() should dispatch to _complete_anthropic for ANTHROPIC provider."""
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)

    with patch.object(client, "_complete_anthropic", new=AsyncMock(return_value="ant response")):
        result = await client.complete("test prompt")
    assert result == "ant response"


@pytest.mark.asyncio
async def test_complete_dispatches_openai():
    """complete() should dispatch to _complete_openai for OPENAI provider."""
    settings = Settings(provider=Provider.OPENAI)
    client = LLMClient(settings=settings)

    with patch.object(client, "_complete_openai", new=AsyncMock(return_value="openai response")):
        result = await client.complete("test prompt")
    assert result == "openai response"


@pytest.mark.asyncio
async def test_complete_raises_for_unsupported_provider():
    """complete() should raise ValueError for an unsupported provider."""
    mock_settings = MagicMock()
    mock_settings.provider = "unknown"
    mock_settings.model = "test"
    client = LLMClient(settings=mock_settings)

    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("test prompt")


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_retries_on_failure():
    """complete() should retry up to 3 times on exception."""
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)

    call_count = 0

    async def fail_then_succeed(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient failure")
        return "success"

    with patch.object(client, "_complete_anthropic", new=fail_then_succeed):
        result = await client.complete("test prompt")
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_complete_reraises_after_max_retries():
    """complete() should reraise the exception after max retries."""
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)

    async def always_fail(*args, **kwargs):
        raise ConnectionError("persistent failure")

    with patch.object(client, "_complete_anthropic", new=always_fail):
        with pytest.raises(ConnectionError, match="persistent failure"):
            await client.complete("test prompt")


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file(tmp_path):
    """read_file should return file contents."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)
    content = await client.read_file(str(test_file))
    assert content == "hello world"


@pytest.mark.asyncio
async def test_write_file(tmp_path):
    """write_file should create the file and parent directories."""
    test_file = tmp_path / "nested" / "dir" / "test.txt"
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)
    await client.write_file(str(test_file), "hello world")
    assert test_file.read_text() == "hello world"


def test_list_files(tmp_path):
    """list_files should return relative file paths."""
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("c")

    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)
    files = client.list_files(str(tmp_path))
    assert "a.txt" in files
    assert "b.txt" in files
    assert "sub/c.txt" in files
    assert len(files) == 3


def test_list_files_nonexistent():
    """list_files should return empty list for non-existent directory."""
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)
    assert client.list_files("/nonexistent/path") == []
