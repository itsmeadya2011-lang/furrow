from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def _settings(provider: Provider) -> Settings:
    return Settings(
        provider=provider,
        anthropic_api_key="test-anthropic",
        openai_api_key="test-openai",
        ollama_base_url="http://localhost:11434",
    )


async def test_complete_routes_to_anthropic():
    client = LLMClient(settings=_settings(Provider.ANTHROPIC))
    with patch.object(client, "_complete_anthropic", new=AsyncMock(return_value="anthropic-response")) as mock_anthropic:
        result = await client.complete("hello", system="sys", model="claude-x")
    assert result == "anthropic-response"
    mock_anthropic.assert_awaited_once_with("hello", "sys", "claude-x")


async def test_complete_routes_to_openai():
    client = LLMClient(settings=_settings(Provider.OPENAI))
    with patch.object(client, "_complete_openai", new=AsyncMock(return_value="openai-response")) as mock_openai:
        result = await client.complete("hello", system="sys", model="gpt-x")
    assert result == "openai-response"
    mock_openai.assert_awaited_once_with("hello", "sys", "gpt-x")


async def test_complete_routes_to_ollama():
    client = LLMClient(settings=_settings(Provider.OLLAMA))
    with patch.object(client, "_complete_ollama", new=AsyncMock(return_value="ollama-response")) as mock_ollama:
        result = await client.complete("hello", system="sys", model="llama-x")
    assert result == "ollama-response"
    mock_ollama.assert_awaited_once_with("hello", "sys", "llama-x")


async def test_complete_unsupported_provider_raises():
    client = LLMClient(settings=_settings(Provider.ANTHROPIC))
    client.settings.provider = "bogus"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        await client.complete("hello")


async def test_read_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    client = LLMClient(settings=_settings(Provider.ANTHROPIC))
    content = await client.read_file(f)
    assert content == "hello world"


async def test_write_file(tmp_path):
    f = tmp_path / "subdir" / "out.txt"
    client = LLMClient(settings=_settings(Provider.ANTHROPIC))
    await client.write_file(f, "written content")
    assert f.read_text() == "written content"


def test_list_files_returns_relative_paths(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y")
    client = LLMClient(settings=_settings(Provider.ANTHROPIC))
    files = client.list_files(tmp_path)
    assert "a.py" in files
    assert str(Path("sub") / "b.py") in files


def test_list_files_returns_empty_for_missing_dir(tmp_path):
    client = LLMClient(settings=_settings(Provider.ANTHROPIC))
    assert client.list_files(tmp_path / "nonexistent") == []


async def test_ollama_complete_uses_httpx_post():
    settings = _settings(Provider.OLLAMA)
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "ollama result"}
    mock_response.raise_for_status = MagicMock()

    with patch("furrow.llm.httpx.AsyncClient") as MockClient:
        mock_instance = MockClient.return_value.__aenter__.return_value
        mock_instance.post = AsyncMock(return_value=mock_response)
        result = await client._complete_ollama("prompt text", "system prompt", "llama3")
    assert result == "ollama result"
    mock_instance.post.assert_awaited_once()
    call_args = mock_instance.post.call_args
    assert call_args.args[0] == "http://localhost:11434/api/generate"
    assert call_args.kwargs["json"]["model"] == "llama3"
    assert call_args.kwargs["json"]["prompt"] == "prompt text"
    assert call_args.kwargs["json"]["system"] == "system prompt"
    assert call_args.kwargs["json"]["stream"] is False