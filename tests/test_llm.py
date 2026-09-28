"""Tests for LLMClient: provider dispatch, Ollama support, file I/O."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def anthropic_settings():
    return Settings(provider=Provider.ANTHROPIC, model="claude-3-5-sonnet")


@pytest.fixture
def ollama_settings():
    return Settings(
        provider=Provider.OLLAMA,
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1",
    )


@pytest.fixture
def openai_settings():
    return Settings(provider=Provider.OPENAI, model="gpt-4o")


class TestCompleteDispatch:
    @pytest.mark.asyncio
    async def test_dispatches_to_anthropic(self, anthropic_settings):
        client = LLMClient(settings=anthropic_settings)
        with patch.object(
            LLMClient, "_complete_anthropic", new_callable=AsyncMock, return_value="claude response"
        ) as mock_method:
            result = await client.complete("hello", system="sys", model="claude-3-5-sonnet")
            assert result == "claude response"
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatches_to_openai(self, openai_settings):
        client = LLMClient(settings=openai_settings)
        with patch.object(
            LLMClient, "_complete_openai", new_callable=AsyncMock, return_value="openai response"
        ) as mock_method:
            result = await client.complete("hello", system="sys")
            assert result == "openai response"

    @pytest.mark.asyncio
    async def test_dispatches_to_ollama(self, ollama_settings):
        client = LLMClient(settings=ollama_settings)
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "message": {"content": "ollama response"}
        }
        fake_response.raise_for_status = MagicMock()
        mock_http = MagicMock()
        mock_http.post = AsyncMock(return_value=fake_response)
        client._http = mock_http
        result = await client.complete("hello", system="sys")
        assert result == "ollama response"

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        settings = Settings(provider=Provider.ANTHROPIC)
        client = LLMClient(settings=settings)
        with patch.object(client, "settings", new=MagicMock(provider=MagicMock(value="unknown"))):
            with pytest.raises(ValueError, match="Unsupported provider"):
                await client.complete("hello")


class TestOllamaModel:
    @pytest.mark.asyncio
    async def test_uses_ollama_model_setting(self, ollama_settings):
        client = LLMClient(settings=ollama_settings)
        fake_response = MagicMock()
        fake_response.json.return_value = {"message": {"content": "ok"}}
        fake_response.raise_for_status = MagicMock()
        mock_http = MagicMock()
        mock_http.post = AsyncMock(return_value=fake_response)
        client._http = mock_http

        await client._complete_ollama("prompt", "", "llama3.1")
        call_args = mock_http.post.call_args
        payload = call_args[1]["json"]
        assert payload["model"] == "llama3.1"
        mock_http.post.assert_called_once()


class TestFileIO:
    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tmp_path, ollama_settings):
        client = LLMClient(settings=ollama_settings)
        with pytest.raises(FileNotFoundError, match="File not found"):
            await client.read_file(tmp_path / "nonexistent.txt")

    @pytest.mark.asyncio
    async def test_read_file_existing(self, tmp_path, ollama_settings):
        client = LLMClient(settings=ollama_settings)
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        content = await client.read_file(f)
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_write_file_creates_parent_dirs(self, tmp_path, ollama_settings):
        client = LLMClient(settings=ollama_settings)
        path = tmp_path / "sub" / "dir" / "file.txt"
        await client.write_file(path, "content")
        assert path.read_text() == "content"

    def test_list_files_nonexistent(self, tmp_path, ollama_settings):
        client = LLMClient(settings=ollama_settings)
        result = client.list_files(tmp_path / "nonexistent")
        assert result == []

    def test_list_files_existing(self, tmp_path, ollama_settings):
        client = LLMClient(settings=ollama_settings)
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        result = client.list_files(tmp_path)
        assert "a.py" in result
        assert "b.py" in result
