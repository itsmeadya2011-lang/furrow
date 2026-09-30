"""Tests for furrow.llm — provider routing, OllamaClient, and FileManager."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import FileManager, LLMClient, OllamaClient

# ------------------------------------------------------------------ #
# Provider routing
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_complete_routes_to_anthropic(monkeypatch: pytest.MonkeyPatch):
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient(settings=settings)

    mock = MagicMock()
    mock.messages.create = AsyncMock()
    mock.messages.create.return_value.content = [MagicMock(text="claude says hi")]
    monkeypatch.setattr(client, "_anthropic", mock, raising=False)

    result = await client.complete("hello", model="claude-3")
    assert result == "claude says hi"


@pytest.mark.asyncio
async def test_complete_routes_to_openai(monkeypatch: pytest.MonkeyPatch):
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    mock = MagicMock()
    mock.chat.completions.create = AsyncMock()
    mock.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="openai says hi"))
    ]
    monkeypatch.setattr(client, "_openai", mock, raising=False)

    result = await client.complete("hello", model="gpt-4")
    assert result == "openai says hi"


@pytest.mark.asyncio
async def test_complete_routes_to_ollama(monkeypatch: pytest.MonkeyPatch):
    """OLLAMA provider should route to OllamaClient instead of raising."""
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)

    mock_ollama = MagicMock()
    mock_ollama.complete = AsyncMock(return_value="ollama says hi")
    monkeypatch.setattr(client, "_ollama", mock_ollama, raising=False)

    result = await client.complete("hello", model="llama3")
    assert result == "ollama says hi"
    mock_ollama.complete.assert_awaited_once_with("hello", "", "llama3")


@pytest.mark.asyncio
async def test_complete_uses_default_model(monkeypatch: pytest.MonkeyPatch):
    """When model=None, falls back to settings.model."""
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    settings.model = "gpt-4o"
    client = LLMClient(settings=settings)

    mock = MagicMock()
    mock.chat.completions.create = AsyncMock()
    mock.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="hi"))
    ]
    monkeypatch.setattr(client, "_openai", mock, raising=False)

    await client.complete("hello")
    call_kwargs = mock.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_complete_with_system_prompt(monkeypatch: pytest.MonkeyPatch):
    """The system prompt should be passed through to the provider."""
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    mock = MagicMock()
    mock.chat.completions.create = AsyncMock()
    mock.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="hi"))
    ]
    monkeypatch.setattr(client, "_openai", mock, raising=False)

    await client.complete("hello", system="be helpful")
    messages = mock.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "be helpful"


# ------------------------------------------------------------------ #
# OllamaClient
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_ollama_client_complete():
    """OllamaClient.complete should POST to /api/generate and return the response field."""
    client = OllamaClient(base_url="http://localhost:11434")
    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"response": "ollama says hi"})
    mock_response.raise_for_status = MagicMock()
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_response)

    with patch.object(type(client), "http", new_callable=lambda: mock_http):
        result = await client.complete("hello", system="be helpful", model="llama3")

    assert result == "ollama says hi"
    mock_http.post.assert_awaited_once()
    call_args = mock_http.post.call_args
    assert call_args[0][0] == "/api/generate"
    body = call_args[1]["json"]
    assert body["prompt"] == "hello"
    assert body["model"] == "llama3"
    assert body["system"] == "be helpful"
    assert body["stream"] is False


@pytest.mark.asyncio
async def test_ollama_client_default_model():
    """When no model is specified, defaults to 'llama3'."""
    client = OllamaClient(base_url="http://localhost:11434")
    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"response": "ok"})
    mock_response.raise_for_status = MagicMock()
    mock_http = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_response)

    with patch.object(type(client), "http", new_callable=lambda: mock_http):
        await client.complete("hello")

    body = mock_http.post.call_args[1]["json"]
    assert body["model"] == "llama3"


@pytest.mark.asyncio
async def test_ollama_client_close():
    """close() should close the underlying HTTP client."""
    client = OllamaClient(base_url="http://localhost:11434")
    mock_http = MagicMock()
    mock_http.aclose = AsyncMock()
    client._http = mock_http  # set directly so close() sees it
    await client.close()
    mock_http.aclose.assert_awaited_once()
    assert client._http is None


# ------------------------------------------------------------------ #
# FileManager
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_file_manager_write_and_read(tmp_path: Path):
    fm = FileManager()
    target = tmp_path / "subdir" / "file.txt"
    await fm.write_file(target, "hello world")
    content = await fm.read_file(target)
    assert content == "hello world"


@pytest.mark.asyncio
async def test_file_manager_write_creates_parents(tmp_path: Path):
    fm = FileManager()
    target = tmp_path / "deeply" / "nested" / "path" / "file.txt"
    await fm.write_file(target, "data")
    assert target.exists()
    assert await fm.read_file(target) == "data"


def test_file_manager_list_files(tmp_path: Path):
    fm = FileManager()
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.py").write_text("b")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("c")
    files = fm.list_files(tmp_path)
    assert "a.txt" in files
    assert "b.py" in files
    assert str(Path("sub") / "c.txt") in files
    assert len(files) == 3


def test_file_manager_list_files_nonexistent():
    fm = FileManager()
    assert fm.list_files("/nonexistent/path/xyz") == []
