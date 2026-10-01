"""Tests for furrow.llm module."""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def tmp_settings(tmp_path):
    """Settings pointed at a temp workspace."""
    return Settings(workspace=tmp_path)


@pytest.fixture
def client(tmp_settings):
    """LLMClient with temp workspace settings."""
    return LLMClient(settings=tmp_settings)


# --- File operations ---

async def test_read_write_file(client, tmp_path):
    """write_file then read_file round-trips correctly."""
    test_file = tmp_path / "example.txt"
    await client.write_file(test_file, "hello world")
    content = await client.read_file(test_file)
    assert content == "hello world"


async def test_write_file_creates_parent_dirs(client, tmp_path):
    """write_file creates parent directories as needed."""
    test_file = tmp_path / "subdir" / "nested" / "file.txt"
    await client.write_file(test_file, "nested content")
    content = await client.read_file(test_file)
    assert content == "nested content"


def test_list_files(client, tmp_path):
    """list_files returns relative paths of files in a directory."""
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "c.txt").write_text("c")

    files = client.list_files(tmp_path)
    assert "a.txt" in files
    assert "b.txt" in files
    assert any(f.endswith("c.txt") for f in files)


def test_list_files_nonexistent(client):
    """list_files returns empty list for nonexistent directory."""
    result = client.list_files("/nonexistent/path/that/does/not/exist")
    assert result == []


def test_list_files_empty_dir(client, tmp_path):
    """list_files returns empty list for an empty directory."""
    assert client.list_files(tmp_path) == []


# --- Provider dispatch ---

def test_ollama_client_uses_base_url():
    """Ollama provider uses ollama_base_url for the client."""
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://custom-host:1234")
    c = LLMClient(settings=s)
    assert str(c.ollama.base_url) == "http://custom-host:1234"


async def test_complete_ollama_dispatch():
    """complete() routes to _complete_ollama for Ollama provider."""
    s = Settings(provider=Provider.OLLAMA)
    c = LLMClient(settings=s)
    with patch.object(c, "_complete_ollama", new_callable=AsyncMock) as mock:
        mock.return_value = "ollama response"
        result = await c.complete("test prompt")
        mock.assert_called_once()
        assert result == "ollama response"


async def test_complete_anthropic_dispatch():
    """complete() routes to _complete_anthropic for Anthropic provider."""
    s = Settings(provider=Provider.ANTHROPIC)
    c = LLMClient(settings=s)
    with patch.object(c, "_complete_anthropic", new_callable=AsyncMock) as mock:
        mock.return_value = "anthropic response"
        result = await c.complete("test prompt")
        mock.assert_called_once()
        assert result == "anthropic response"


async def test_complete_openai_dispatch():
    """complete() routes to _complete_openai for OpenAI provider."""
    s = Settings(provider=Provider.OPENAI)
    c = LLMClient(settings=s)
    with patch.object(c, "_complete_openai", new_callable=AsyncMock) as mock:
        mock.return_value = "openai response"
        result = await c.complete("test prompt")
        mock.assert_called_once()
        assert result == "openai response"


async def test_complete_unsupported_provider():
    """complete() raises ValueError for unknown providers."""
    s = MagicMock(spec=Settings)
    s.provider = "custom"
    s.model = "test-model"
    s.ollama_base_url = "http://localhost:11434"
    c = LLMClient(settings=s)
    with pytest.raises(ValueError, match="Unsupported provider"):
        await c.complete("test")


async def test_complete_uses_default_model():
    """complete() uses settings.model when no model is provided."""
    s = Settings(provider=Provider.ANTHROPIC)
    c = LLMClient(settings=s)
    with patch.object(c, "_complete_anthropic", new_callable=AsyncMock) as mock:
        mock.return_value = "ok"
        await c.complete("prompt")
        args, kwargs = mock.call_args
        assert args[2] == s.model  # third positional arg is model
