"""Tests for the LLMClient — provider dispatch, retry logic, and OLLAMA support."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import DEFAULT_MAX_RETRIES, LLMClient


def _settings(provider: Provider = Provider.OPENAI, **kwargs) -> Settings:
    """Build a Settings instance with test-friendly defaults."""
    data = {
        "provider": provider.value,
        "openai_api_key": "sk-test-123",
        "anthropic_api_key": "sk-ant-test",
        "ollama_base_url": "http://localhost:11434",
        "max_parallel_tasks": 5,
        "max_cycles": 0,
        "max_retries": 3,
        "request_timeout": 30.0,
    }
    data.update(kwargs)
    return Settings(**data)


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_unsupported_provider():
    settings = _settings(provider=Provider.OPENAI)
    # Patch provider with a plain string to simulate an unsupported value.
    with patch.object(settings, "provider", "custom"):
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("hello")


@pytest.mark.asyncio
async def test_complete_uses_default_model():
    settings = _settings(provider=Provider.OPENAI)
    client = LLMClient(settings=settings)
    client._openai = AsyncMock()
    client._openai.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="response"))]
    )
    result = await client.complete("hello")
    assert result == "response"
    # Should use settings.model by default.
    call_kwargs = client._openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == settings.model


@pytest.mark.asyncio
async def test_complete_uses_overridden_model():
    settings = _settings(provider=Provider.OPENAI)
    client = LLMClient(settings=settings)
    client._openai = AsyncMock()
    client._openai.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="response"))]
    )
    result = await client.complete("hello", model="gpt-4")
    assert result == "response"
    call_kwargs = client._openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# OLLAMA provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_ollama_provider():
    settings = _settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    client._httpx = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "ollama reply"}
    client._httpx.post.return_value = mock_resp

    result = await client.complete("hi", model="llama3")
    assert result == "ollama reply"
    client._httpx.post.assert_awaited_once()
    args, kwargs = client._httpx.post.call_args
    assert "/api/generate" in args[0]
    assert kwargs["json"]["model"] == "llama3"
    assert kwargs["json"]["prompt"] == "hi"


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_on_timeout_then_succeeds():
    """Should retry on timeout and eventually succeed."""
    settings = _settings(provider=Provider.OPENAI)
    client = LLMClient(settings=settings)
    client._openai = AsyncMock()

    # First call times out, second succeeds.
    timeout_exc = asyncio.TimeoutError("timeout")
    client._openai.chat.completions.create.side_effect = [timeout_exc, MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )]

    result = await client.complete("prompt")
    assert result == "ok"
    assert client._openai.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_retry_exhausted_on_persistent_timeout():
    """Should raise after max retries on persistent errors."""
    settings = _settings(provider=Provider.OPENAI)
    client = LLMClient(settings=settings)
    client._openai = AsyncMock()
    client._openai.chat.completions.create.side_effect = asyncio.TimeoutError("timeout")

    with pytest.raises(asyncio.TimeoutError):
        await client.complete("prompt")
    assert client._openai.chat.completions.create.await_count == DEFAULT_MAX_RETRIES


@pytest.mark.asyncio
async def test_no_retry_on_non_retryable_error():
    """Non-retryable errors (e.g. auth) should not be retried."""
    settings = _settings(provider=Provider.OPENAI)
    client = LLMClient(settings=settings)
    client._openai = AsyncMock()
    # An arbitrary non-retryable error.
    client._openai.chat.completions.create.side_effect = RuntimeError("auth failed")

    with pytest.raises(RuntimeError, match="auth failed"):
        await client.complete("prompt")
    assert client._openai.chat.completions.create.await_count == 1


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    client = LLMClient(settings=_settings())
    result = await client.read_file(str(f))
    assert result == "hello world"


@pytest.mark.asyncio
async def test_write_file_creates_parent_dir(tmp_path):
    client = LLMClient(settings=_settings())
    target = tmp_path / "sub" / "deep" / "file.txt"
    await client.write_file(str(target), "content")
    assert target.read_text() == "content"


def test_list_files(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "b.txt").write_text("b")
    client = LLMClient(settings=_settings())
    files = client.list_files(str(tmp_path))
    assert "a.txt" in files
    assert "dir/b.txt" in files


def test_list_files_nonexistent():
    client = LLMClient(settings=_settings())
    assert client.list_files("/nonexistent/path") == []
