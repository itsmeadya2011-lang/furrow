from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def _anthropic_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


def _make_settings(provider: Provider) -> Settings:
    return Settings(
        provider=provider,
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
        model="test-model",
    )


def test_complete_dispatches_anthropic():
    settings = _make_settings(Provider.ANTHROPIC)
    client = LLMClient(settings=settings)

    mock_anthropic = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="anthropic result")]
    mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

    with patch.object(LLMClient, "anthropic", new_callable=PropertyMock) as mock_prop:
        mock_prop.return_value = mock_anthropic
        result = asyncio.run(client.complete("hello", system="sys", model="m"))

    assert result == "anthropic result"
    mock_anthropic.messages.create.assert_awaited_once()
    _, kwargs = mock_anthropic.messages.create.call_args
    assert kwargs["model"] == "m"
    assert kwargs["max_tokens"] == 4096


def test_complete_dispatches_openai():
    settings = _make_settings(Provider.OPENAI)
    client = LLMClient(settings=settings)

    mock_openai = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="openai result"))]
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch.object(LLMClient, "openai", new_callable=PropertyMock) as mock_prop:
        mock_prop.return_value = mock_openai
        result = asyncio.run(client.complete("hello", system="sys", model="m"))

    assert result == "openai result"
    mock_openai.chat.completions.create.assert_awaited_once()
    _, kwargs = mock_openai.chat.completions.create.call_args
    assert kwargs["model"] == "m"


def test_complete_dispatches_ollama():
    settings = _make_settings(Provider.OLLAMA)
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"response": "ollama result"})
    mock_response.raise_for_status = MagicMock()

    mock_async_client = MagicMock()
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)
    mock_async_client.post = AsyncMock(return_value=mock_response)

    with patch("furrow.llm.httpx.AsyncClient", return_value=mock_async_client):
        result = asyncio.run(client.complete("hello", system="sys", model="m"))

    assert result == "ollama result"
    mock_async_client.post.assert_awaited_once()
    _, kwargs = mock_async_client.post.call_args
    assert kwargs["json"]["model"] == "m"
    assert kwargs["json"]["prompt"] == "hello"


def test_complete_unsupported_provider():
    settings = _make_settings(Provider.ANTHROPIC)
    settings.provider = "unknown"
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError):
        asyncio.run(client.complete("hello"))


def test_read_file_encoding(tmp_path, monkeypatch):

    file_path = tmp_path / "input.txt"
    file_path.write_text("file contents", encoding="utf-8")

    captured = {}

    class FakeFile:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        async def read(self):
            return "file contents"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    settings = _make_settings(Provider.OLLAMA)
    client = LLMClient(settings=settings)

    with patch("furrow.llm.aiofiles.open", side_effect=FakeFile):
        result = asyncio.run(client.read_file(file_path))

    assert result == "file contents"
    assert captured["kwargs"].get("encoding") == "utf-8"
