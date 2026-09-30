"""Tests for the LLMClient."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


class TestLLMClientProviders:
    def test_provider_anthropic(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")
        client = LLMClient(settings=settings)
        assert client.settings.provider == Provider.ANTHROPIC

    def test_provider_openai(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key="test")
        client = LLMClient(settings=settings)
        assert client.settings.provider == Provider.OPENAI

    def test_provider_ollama(self):
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        assert client.settings.provider == Provider.OLLAMA

    def test_defaults_to_global_settings(self):
        """LLMClient() should use the global default settings."""
        client = LLMClient()
        assert client.settings is not None  # uses global settings


class TestLLMClientAnthropic:
    @pytest.mark.asyncio
    async def test_complete_anthropic(self):
        settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
            model="claude-3-5-sonnet",
        )
        client = LLMClient(settings=settings)
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="hello world")]
        client._anthropic = MagicMock()
        client._anthropic.messages.create = AsyncMock(return_value=mock_response)

        result = await client.complete("hello")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_complete_no_api_key_raises(self):
        settings = Settings(provider=Provider.ANTHROPIC)
        # Temporarily remove env var
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            client = LLMClient(settings=settings)
            client._anthropic = None
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                _ = client.anthropic


class TestLLMClientOpenAI:
    @pytest.mark.asyncio
    async def test_complete_openai(self):
        settings = Settings(
            provider=Provider.OPENAI,
            openai_api_key="test-key",
            model="gpt-4",
        )
        client = LLMClient(settings=settings)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="openai response"))]
        client._openai = MagicMock()
        client._openai.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.complete("hello")
        assert result == "openai response"


class TestLLMClientOllama:
    @pytest.mark.asyncio
    async def test_complete_ollama(self):
        settings = Settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434",
            model="llama3",
        )
        client = LLMClient(settings=settings)
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"message": {"content": "ollama response"}})
        mock_response.raise_for_status = MagicMock()
        client._httpx = MagicMock()
        client._httpx.post = AsyncMock(return_value=mock_response)

        result = await client.complete("hello")
        assert result == "ollama response"

    @pytest.mark.asyncio
    async def test_ollama_uses_configured_url(self):
        settings = Settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://custom-host:1234",
        )
        client = LLMClient(settings=settings)
        # The httpx client should use the configured base URL
        httpx_client = client.httpx_client
        assert httpx_client.base_url == "http://custom-host:1234"


class TestLLMClientFileOps:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello file content")
        client = LLMClient(settings=Settings())
        content = await client.read_file(str(test_file))
        assert content == "hello file content"

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        test_file = tmp_path / "subdir" / "output.txt"
        client = LLMClient(settings=Settings())
        await client.write_file(str(test_file), "new content")
        assert test_file.exists()
        assert test_file.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_write_file_creates_parent_dirs(self, tmp_path):
        test_file = tmp_path / "a" / "b" / "c" / "deep.txt"
        client = LLMClient(settings=Settings())
        await client.write_file(str(test_file), "deep content")
        assert test_file.exists()

    def test_list_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.txt").write_text("c")
        client = LLMClient(settings=Settings())
        files = client.list_files(str(tmp_path))
        assert "a.txt" in files
        assert "b.txt" in files
        assert "sub/c.txt" in files

    def test_list_files_nonexistent(self):
        client = LLMClient(settings=Settings())
        files = client.list_files("/nonexistent/path")
        assert files == []

    def test_list_files_empty_dir(self, tmp_path):
        client = LLMClient(settings=Settings())
        files = client.list_files(str(tmp_path))
        assert files == []

    @pytest.mark.asyncio
    async def test_close_releases_httpx(self):
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        # Access the client to initialize it
        _ = client.httpx_client
        assert client._httpx is not None
        await client.close()
        assert client._httpx is None

    @pytest.mark.asyncio
    async def test_close_without_httpx(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="key")
        client = LLMClient(settings=settings)
        # close() should be a no-op if _httpx was never initialized
        await client.close()
        assert client._httpx is None


class TestLLMClientRetry:
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """LLMClient should retry on transient errors."""
        import httpx as httpx_mod

        settings = Settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434",
            model="llama3",
        )
        client = LLMClient(settings=settings)

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"message": {"content": "success"}})
        mock_response.raise_for_status = MagicMock()

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx_mod.ConnectError("connection refused")
            return mock_response

        client._httpx = MagicMock()
        client._httpx.post = mock_post

        result = await client.complete("hello")
        assert result == "success"
        assert call_count == 3
