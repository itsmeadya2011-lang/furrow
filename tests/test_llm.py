"""Tests for the LLM client, including the Ollama provider and retry logic."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


# ---------------------------------------------------------------------------
# Ollama provider tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_provider_sends_correct_payload():
    """Ollama provider should send the correct payload to the API."""
    settings = Settings(provider=Provider.OLLAMA, model="llama3")
    client = LLMClient(settings=settings)

    mock_response = {"response": "Hello from Ollama"}
    mock_httpx = MagicMock()
    mock_httpx.post = AsyncMock(
        return_value=MagicMock(
            json=MagicMock(return_value=mock_response),
            raise_for_status=MagicMock(),
        )
    )
    client._httpx = mock_httpx

    result = await client.complete("Hello", model="llama3")

    assert result == "Hello from Ollama"

    call_kwargs = mock_httpx.post.call_args
    payload = call_kwargs.kwargs["json"]
    assert payload["model"] == "llama3"
    assert payload["prompt"] == "Hello"
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == settings.max_tokens
    assert payload["options"]["temperature"] == settings.temperature


@pytest.mark.asyncio
async def test_ollama_provider_uses_settings_model():
    """Ollama provider should use settings.model when no model is specified."""
    settings = Settings(provider=Provider.OLLAMA, model="my-llama2")
    client = LLMClient(settings=settings)

    mock_httpx = MagicMock()
    mock_httpx.post = AsyncMock(
        return_value=MagicMock(
            json=MagicMock(return_value={"response": "ok"}),
            raise_for_status=MagicMock(),
        )
    )
    client._httpx = mock_httpx

    await client.complete("test")

    call_kwargs = mock_httpx.post.call_args
    assert call_kwargs.kwargs["json"]["model"] == "my-llama2"


@pytest.mark.asyncio
async def test_unsupported_provider_raises():
    """complete() should raise ValueError for unknown providers."""
    settings = Settings(provider=Provider.ANTHROPIC, model="test")
    client = LLMClient(settings=settings)

    with patch.object(client.settings, "provider", "invalid"):
        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("hello")


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_on_failure_then_success():
    """complete() should retry on failure and succeed on the third attempt."""
    settings = Settings(provider=Provider.OLLAMA, model="llama3")
    client = LLMClient(settings=settings)

    call_count = 0

    async def _fail_twice_then_succeed(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.HTTPError("Simulated network error")
        return MagicMock(
            json=MagicMock(return_value={"response": "success"}),
            raise_for_status=MagicMock(),
        )

    mock_httpx = MagicMock()
    mock_httpx.post = _fail_twice_then_succeed
    client._httpx = mock_httpx

    result = await client.complete("test prompt")

    assert result == "success"
    assert call_count == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_retry_exhausted_raises():
    """complete() should raise the last exception after max retries."""
    settings = Settings(provider=Provider.OLLAMA, model="llama3")
    client = LLMClient(settings=settings)

    mock_httpx = MagicMock()
    mock_httpx.post = AsyncMock(side_effect=httpx.HTTPError("Always fails"))
    client._httpx = mock_httpx

    with pytest.raises(httpx.HTTPError):
        await client.complete("test prompt")


# ---------------------------------------------------------------------------
# File utility tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file():
    """LLMClient.read_file should read file contents."""
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))

    content = await client.read_file("pyproject.toml")
    assert "furrow" in content


@pytest.mark.asyncio
async def test_list_files():
    """LLMClient.list_files should return files in a directory."""
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))

    files = client.list_files("furrow")
    assert len(files) > 0
    assert any("config.py" in f for f in files)
