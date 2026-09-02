from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from furrow.config import Provider
from furrow.llm import LLMClient


def test_list_files_excludes_dirs(tmp_path) -> None:
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("z")

    client = LLMClient()
    files = client.list_files(tmp_path)
    assert "a.py" in files
    assert "b.txt" in files
    assert "sub/c.py" in files
    assert all(not f.endswith("sub") for f in files)


def test_list_files_missing_directory(tmp_path) -> None:
    client = LLMClient()
    assert client.list_files(tmp_path / "nope") == []


@pytest.mark.asyncio
async def test_complete_dispatches_anthropic(monkeypatch) -> None:
    client = LLMClient()
    monkeypatch.setattr(client.settings, "provider", Provider.ANTHROPIC)

    mock_resp = AsyncMock()
    mock_resp.content = [AsyncMock(text="hello from anthropic")]
    mock_messages = AsyncMock()
    mock_messages.create = AsyncMock(return_value=mock_resp)
    client._anthropic = mock_messages

    out = await client._complete_anthropic("hi", "", "claude-x")
    assert out == "hello from anthropic"
    mock_messages.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_complete_dispatches_openai(monkeypatch) -> None:
    client = LLMClient()
    monkeypatch.setattr(client.settings, "provider", Provider.OPENAI)

    mock_choice = AsyncMock()
    mock_choice.message.content = "hello from openai"
    mock_resp = AsyncMock()
    mock_resp.choices = [mock_choice]
    mock_chat = AsyncMock()
    mock_chat.completions.create = AsyncMock(return_value=mock_resp)
    client._openai = mock_chat

    out = await client._complete_openai("hi", "", "gpt-x")
    assert out == "hello from openai"
    mock_chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_complete_dispatches_ollama(monkeypatch) -> None:
    client = LLMClient()
    monkeypatch.setattr(client.settings, "provider", Provider.OLLAMA)

    mock_resp = AsyncMock()
    mock_resp.raise_for_status = lambda: None
    mock_resp.json = lambda: {"message": {"content": "hi"}}
    mock_post = AsyncMock(return_value=mock_resp)
    client._ollama = AsyncMock(post=mock_post)

    out = await client._complete_ollama("hi", "", "llama3")
    assert out == "hi"
    mock_post.assert_awaited_once()


@pytest.mark.asyncio
async def test_complete_unsupported_raises(monkeypatch) -> None:
    client = LLMClient()

    class FakeProvider:
        pass

    fake = FakeProvider()
    monkeypatch.setattr(client.settings, "provider", fake)
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hi")