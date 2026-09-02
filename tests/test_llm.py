"""Tests for LLMClient."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


# ---------------------------------------------------------------------------
# Provider properties
# ---------------------------------------------------------------------------


class TestProviderProperties:
    def test_anthropic_client_raises_without_key(self, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        settings = Settings(anthropic_api_key=None)
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            _ = client.anthropic

    def test_openai_client_raises_without_key(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = Settings(openai_api_key=None)
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            _ = client.openai

    def test_ollama_client_initializes(self) -> None:
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        assert client.ollama is not None
        assert client._ollama is not None

    def test_anthropic_client_caching(self) -> None:
        settings = Settings(anthropic_api_key="test-key")
        client = LLMClient(settings=settings)
        c1 = client.anthropic
        c2 = client.anthropic
        assert c1 is c2  # type: ignore

    def test_ollama_uses_custom_base_url(self) -> None:
        settings = Settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:9999",
        )
        client = LLMClient(settings=settings)
        # The AsyncClient should be configured with the custom base URL
        assert client.ollama is not None


# ---------------------------------------------------------------------------
# complete() routing
# ---------------------------------------------------------------------------


class TestCompleteRouting:
    @pytest.mark.asyncio
    async def test_complete_routes_to_anthropic(self) -> None:
        settings = Settings(anthropic_api_key="test-key", provider=Provider.ANTHROPIC)
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock:
            mock.return_value = "response"
            result = await client.complete("hello", model="claude-test")
        mock.assert_awaited_once_with("hello", "", "claude-test")
        assert result == "response"

    @pytest.mark.asyncio
    async def test_complete_routes_to_openai(self) -> None:
        settings = Settings(openai_api_key="test-key", provider=Provider.OPENAI)
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_openai", new_callable=AsyncMock) as mock:
            mock.return_value = "response"
            result = await client.complete("hello", system="sys", model="gpt-test")
        mock.assert_awaited_once_with("hello", "sys", "gpt-test")
        assert result == "response"

    @pytest.mark.asyncio
    async def test_complete_routes_to_ollama(self) -> None:
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = "response"
            result = await client.complete("hello", model="llama-test")
        mock.assert_awaited_once_with("hello", "", "llama-test")
        assert result == "response"

    @pytest.mark.asyncio
    async def test_complete_uses_default_model(self) -> None:
        settings = Settings(anthropic_api_key="test-key", model="default-model")
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock:
            mock.return_value = "response"
            await client.complete("hello")
        mock.assert_awaited_once_with("hello", "", "default-model")

    @pytest.mark.asyncio
    async def test_complete_sends_system_prompt(self) -> None:
        settings = Settings(anthropic_api_key="test-key", provider=Provider.ANTHROPIC)
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock:
            mock.return_value = "response"
            result = await client.complete("hello", system="be helpful", model="claude-test")
        mock.assert_awaited_once_with("hello", "be helpful", "claude-test")
        assert result == "response"


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------


class TestChat:
    @pytest.mark.asyncio
    async def test_chat_anthropic_separates_system(self) -> None:
        settings = Settings(anthropic_api_key="test-key", provider=Provider.ANTHROPIC)
        client = LLMClient(settings=settings)
        messages = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
        with patch.object(client, "_chat_anthropic", new_callable=AsyncMock) as mock:
            mock.return_value = "reply"
            result = await client.chat(messages, model="claude-test")
        mock.assert_awaited_once_with(
            [{"role": "user", "content": "hi"}],
            "be helpful",
            "claude-test",
        )
        assert result == "reply"

    @pytest.mark.asyncio
    async def test_chat_openai_passes_messages(self) -> None:
        settings = Settings(openai_api_key="test-key", provider=Provider.OPENAI)
        client = LLMClient(settings=settings)
        messages = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
        with patch.object(client, "_chat_openai", new_callable=AsyncMock) as mock:
            mock.return_value = "reply"
            await client.chat(messages, model="gpt-test")
        mock.assert_awaited_once_with(messages, "gpt-test")

    @pytest.mark.asyncio
    async def test_chat_ollama_uses_ollama_endpoint(self) -> None:
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        messages = [{"role": "user", "content": "hi"}]
        with patch.object(client, "_chat_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = "reply"
            await client.chat(messages, model="llama-test")
        mock.assert_awaited_once_with(messages, "llama-test")


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


class TestFileOperations:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        client = LLMClient(settings=Settings(anthropic_api_key="key"))
        content = await client.read_file(f)
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_write_file_creates_dirs(self, tmp_path) -> None:
        f = tmp_path / "subdir" / "output.txt"
        client = LLMClient(settings=Settings(anthropic_api_key="key"))
        await client.write_file(f, "content here")
        assert f.read_text() == "content here"
        assert f.parent.exists()

    @pytest.mark.asyncio
    async def test_list_files_recursive(self, tmp_path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("b")
        client = LLMClient(settings=Settings(anthropic_api_key="key"))
        files = await client.list_files(tmp_path)
        assert "a.txt" in files
        assert str(Path("sub") / "b.py") in files or "sub/b.py" in files

    @pytest.mark.asyncio
    async def test_list_files_nonexistent(self) -> None:
        client = LLMClient(settings=Settings(anthropic_api_key="key"))
        files = await client.list_files("/nonexistent/path")
        assert files == []
