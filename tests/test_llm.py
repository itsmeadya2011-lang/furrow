from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest
from openai import APIStatusError as OpenAIAPIStatusError

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def _anthropic_response(text: str) -> MagicMock:
    content = MagicMock()
    content.text = text
    resp = MagicMock()
    resp.content = [content]
    return resp


def _openai_response(text: str) -> MagicMock:
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _ollama_httpx_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


async def test_complete_anthropic(mock_settings: Settings) -> None:
    client = LLMClient(settings=mock_settings)
    client._anthropic = AsyncMock()
    client._anthropic.messages.create = AsyncMock(
        return_value=_anthropic_response("hello world")
    )

    result = await client.complete("hi")

    assert result == "hello world"
    client._anthropic.messages.create.assert_awaited_once()
    _, kwargs = client._anthropic.messages.create.call_args
    assert kwargs["model"] == "test-model"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


async def test_complete_openai(mock_settings: Settings) -> None:
    mock_settings.provider = Provider.OPENAI
    client = LLMClient(settings=mock_settings)
    client._openai = AsyncMock()
    client._openai.chat.completions.create = AsyncMock(
        return_value=_openai_response("openai reply")
    )

    result = await client.complete("hi")

    assert result == "openai reply"
    client._openai.chat.completions.create.assert_awaited_once()
    _, kwargs = client._openai.chat.completions.create.call_args
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"][1] == {"role": "user", "content": "hi"}


async def test_complete_ollama(mock_settings: Settings) -> None:
    mock_settings.provider = Provider.OLLAMA
    mock_settings.ollama_base_url = "http://localhost:11434"
    client = LLMClient(settings=mock_settings)

    http_response = _ollama_httpx_response({"message": {"content": "ollama reply"}})
    inner_client = AsyncMock()
    inner_client.post = AsyncMock(return_value=http_response)
    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=inner_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)

    with patch("furrow.llm.httpx.AsyncClient", return_value=fake_client) as AC:
        result = await client.complete("hi")

    assert result == "ollama reply"
    AC.assert_called_once_with(timeout=120.0)
    inner_client.post.assert_awaited_once()
    args, _ = inner_client.post.call_args
    assert args[0] == "http://localhost:11434/api/chat"
    body = inner_client.post.call_args.kwargs["json"]
    assert body["model"] == "test-model"
    assert body["stream"] is False
    assert body["messages"][1] == {"role": "user", "content": "hi"}


async def test_empty_anthropic_response(mock_settings: Settings) -> None:
    client = LLMClient(settings=mock_settings)
    client._anthropic = AsyncMock()

    # Empty content list.
    resp = MagicMock()
    resp.content = []
    client._anthropic.messages.create = AsyncMock(return_value=resp)

    with pytest.raises(ValueError, match="Empty response from Anthropic"):
        await client.complete("hi")


async def test_empty_openai_response(mock_settings: Settings) -> None:
    mock_settings.provider = Provider.OPENAI
    client = LLMClient(settings=mock_settings)
    client._openai = AsyncMock()

    resp = MagicMock()
    resp.choices = []
    client._openai.chat.completions.create = AsyncMock(return_value=resp)

    with pytest.raises(ValueError, match="Empty response from OpenAI"):
        await client.complete("hi")


async def test_empty_ollama_response(mock_settings: Settings) -> None:
    mock_settings.provider = Provider.OLLAMA
    client = LLMClient(settings=mock_settings)

    http_response = _ollama_httpx_response({})
    inner_client = AsyncMock()
    inner_client.post = AsyncMock(return_value=http_response)
    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=inner_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)

    with patch("furrow.llm.httpx.AsyncClient", return_value=fake_client):
        with pytest.raises(ValueError, match="Empty response from Ollama"):
            await client.complete("hi")


async def test_retry_on_api_error(mock_settings: Settings) -> None:
    client = LLMClient(settings=mock_settings)
    client._anthropic = AsyncMock()

    error = anthropic.APIStatusError(
        message="server error",
        response=MagicMock(status_code=500, headers={}),
        body=None,
    )
    create_mock = AsyncMock(side_effect=[error, error, _anthropic_response("recovered")])
    client._anthropic.messages.create = create_mock

    with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
        result = await client.complete("hi")

    assert result == "recovered"
    assert create_mock.await_count == 3


async def test_read_file(mock_settings: Settings, temp_workspace: Path) -> None:
    (temp_workspace / "data.txt").write_text("file contents")
    client = LLMClient(settings=mock_settings)

    result = await client.read_file("data.txt")
    assert result == "file contents"


async def test_write_file(mock_settings: Settings, temp_workspace: Path) -> None:
    client = LLMClient(settings=mock_settings)

    await client.write_file("subdir/nested.txt", "hello\nworld")

    written = (temp_workspace / "subdir" / "nested.txt").read_text()
    assert written == "hello\nworld"


def test_list_files(mock_settings: Settings, temp_workspace: Path) -> None:
    (temp_workspace / "a.txt").write_text("a")
    (temp_workspace / "subdir").mkdir()
    (temp_workspace / "subdir" / "b.txt").write_text("b")

    client = LLMClient(settings=mock_settings)
    files = client.list_files(".")

    assert "a.txt" in files
    assert any("b.txt" in f for f in files)


def test_workspace_relative_paths(mock_settings: Settings, temp_workspace: Path) -> None:
    client = LLMClient(settings=mock_settings)
    resolved = client._resolve_path("foo/bar.txt")
    assert resolved == (temp_workspace / "foo" / "bar.txt")
    assert resolved.is_absolute()


async def test_unsupported_provider(mock_settings: Settings) -> None:
    mock_settings.provider = "bogus"  # type: ignore[assignment]
    client = LLMClient(settings=mock_settings)

    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hi")