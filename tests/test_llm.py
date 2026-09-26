"""Tests for the LLMClient provider dispatch and file I/O."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient

# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_anthropic_provider(mock_settings: Settings):
    """Anthropic provider dispatches to _complete_anthropic."""
    client = LLMClient(settings=mock_settings)
    with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock_anthropic:
        mock_anthropic.return_value = "anthropic response"
        result = await client.complete("prompt", model="claude-test")
    assert result == "anthropic response"
    mock_anthropic.assert_awaited_once_with("prompt", "", "claude-test")


@pytest.mark.asyncio
async def test_complete_openai_provider(openai_settings: Settings):
    """OpenAI provider dispatches to _complete_openai with the openai client."""
    client = LLMClient(settings=openai_settings)
    with patch.object(client, "_complete_openai", new_callable=AsyncMock) as mock_openai:
        mock_openai.return_value = "openai response"
        result = await client.complete("prompt", model="gpt-4o")
    assert result == "openai response"
    mock_openai.assert_awaited_once()
    # Verify the client passed is the OpenAI client, not Ollama
    call_args = mock_openai.call_args
    assert call_args.args[3] is client.openai


@pytest.mark.asyncio
async def test_complete_ollama_provider(ollama_settings: Settings):
    """Ollama provider dispatches to _complete_openai with the ollama client."""
    client = LLMClient(settings=ollama_settings)
    with patch.object(client, "_complete_openai", new_callable=AsyncMock) as mock_openai:
        mock_openai.return_value = "ollama response"
        result = await client.complete("prompt", model="llama3")
    assert result == "ollama response"
    mock_openai.assert_awaited_once()
    # Verify the client passed is the Ollama client
    call_args = mock_openai.call_args
    assert call_args.args[3] is client.ollama


@pytest.mark.asyncio
async def test_unsupported_provider_raises():
    """An unsupported provider raises ValueError."""
    mock_settings = MagicMock()
    mock_settings.provider = "unknown"
    mock_settings.model = "test"
    mock_settings.planner_model = "test"
    mock_settings.worker_model = "test"
    mock_settings.tester_model = "test"
    mock_settings.ollama_base_url = "http://localhost:11434"
    client = LLMClient(settings=mock_settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("prompt")


# ---------------------------------------------------------------------------
# Provider client lazy initialization
# ---------------------------------------------------------------------------

def test_anthropic_client_lazy_init(mock_settings: Settings):
    """Anthropic client is lazily initialized."""
    client = LLMClient(settings=mock_settings)
    assert client._anthropic is None
    a = client.anthropic
    assert client._anthropic is not None
    assert a is not None


def test_openai_client_lazy_init(openai_settings: Settings):
    """OpenAI client is lazily initialized."""
    client = LLMClient(settings=openai_settings)
    assert client._openai is None
    o = client.openai
    assert client._openai is not None
    assert o is not None


def test_ollama_client_lazy_init(ollama_settings: Settings):
    """Ollama client is lazily initialized with custom base URL."""
    client = LLMClient(settings=ollama_settings)
    assert client._ollama is None
    ollama = client.ollama
    assert client._ollama is not None
    assert ollama is not None


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------

def test_anthropic_missing_api_key_raises():
    """Anthropic client raises ValueError when no API key is set."""
    settings = Settings(anthropic_api_key=None, openai_api_key=None)
    with patch.dict(os.environ, {}, clear=True):
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            _ = client.anthropic


def test_openai_missing_api_key_raises():
    """OpenAI client raises ValueError when no API key is set."""
    settings = Settings(
        provider=Provider.OPENAI,
        openai_api_key=None,
        anthropic_api_key=None,
    )
    with patch.dict(os.environ, {}, clear=True):
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            _ = client.openai


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_file(mock_settings: Settings, tmp_path: Path):
    """read_file returns file contents."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")
    client = LLMClient(settings=mock_settings)
    content = await client.read_file(test_file)
    assert content == "hello world"


@pytest.mark.asyncio
async def test_write_file(mock_settings: Settings, tmp_path: Path):
    """write_file creates parent dirs and writes content."""
    client = LLMClient(settings=mock_settings)
    target = tmp_path / "subdir" / "output.txt"
    await client.write_file(target, "file content")
    assert target.exists()
    assert target.read_text() == "file content"


def test_list_files(mock_settings: Settings, tmp_path: Path):
    """list_files returns relative file paths."""
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y")
    client = LLMClient(settings=mock_settings)
    files = client.list_files(tmp_path)
    assert set(files) == {"a.py", "sub/b.py"}


def test_list_files_nonexistent(mock_settings: Settings):
    """list_files returns empty list for nonexistent directory."""
    client = LLMClient(settings=mock_settings)
    assert client.list_files("/nonexistent/path/xyz") == []
