"""Tests for LLMClient — provider routing, file operations, error handling."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


# --- Provider routing ---


class TestProviderRouting:
    async def test_complete_routes_anthropic(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock:
            mock.return_value = "anthropic response"
            result = await client.complete("test prompt")
        assert result == "anthropic response"
        mock.assert_awaited_once()

    async def test_complete_routes_openai(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_openai", new_callable=AsyncMock) as mock:
            mock.return_value = "openai response"
            result = await client.complete("test prompt")
        assert result == "openai response"
        mock.assert_awaited_once()

    async def test_complete_routes_ollama(self):
        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = "ollama response"
            result = await client.complete("test prompt")
        assert result == "ollama response"
        mock.assert_awaited_once()

    async def test_complete_uses_provided_model(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock:
            mock.return_value = "resp"
            await client.complete("test", model="custom-model")
        args, kwargs = mock.call_args
        # _complete_anthropic(prompt, system, model)
        assert args[2] == "custom-model"

    async def test_complete_default_system_prompt(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="t")
        client = LLMClient(settings=settings)
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock:
            mock.return_value = "resp"
            await client.complete("test prompt")
        args, kwargs = mock.call_args
        # _complete_anthropic(prompt, system, model) — system defaults to ""
        assert args[1] == ""


class TestOllamaProperty:
    def test_ollama_uses_configured_base_url(self):
        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://custom:1234")
        client = LLMClient(settings=settings)
        ollama_client = client.ollama
        # The AsyncOpenAI should be created with the custom base URL
        assert ollama_client.base_url == "http://custom:1234"

    def test_ollama_lazy_init(self):
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        assert client._ollama is None
        _ = client.ollama
        assert client._ollama is not None


class TestProviderErrors:
    async def test_complete_unsupported_provider_raises(self):
        mock_settings = MagicMock()
        mock_settings.provider = "unsupported-provider"
        mock_settings.model = "test-model"
        mock_settings.max_cycles = 0
        client = LLMClient(settings=mock_settings)
        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("test")

    def test_anthropic_property_no_api_key_raises(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
        # Clear env var to be sure
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            client = LLMClient(settings=settings)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                _ = client.anthropic

    def test_openai_property_no_api_key_raises(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key=None)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            client = LLMClient(settings=settings)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                _ = client.openai


# --- File operations ---


class TestFileOperations:
    async def test_read_file(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello world")

        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="t")
        client = LLMClient(settings=settings)
        content = await client.read_file(str(filepath))
        assert content == "hello world"

    async def test_write_file_creates_new_file(self, tmp_path):
        filepath = tmp_path / "subdir" / "new.txt"
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="t")
        client = LLMClient(settings=settings)
        await client.write_file(str(filepath), "file content")
        assert filepath.read_text() == "file content"

    async def test_write_file_creates_parent_dirs(self, tmp_path):
        filepath = tmp_path / "a" / "b" / "c" / "deep.txt"
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="t")
        client = LLMClient(settings=settings)
        await client.write_file(str(filepath), "deep content")
        assert filepath.read_text() == "deep content"

    async def test_write_file_overwrites_existing(self, tmp_path):
        filepath = tmp_path / "overwrite.txt"
        filepath.write_text("old content")

        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="t")
        client = LLMClient(settings=settings)
        await client.write_file(str(filepath), "new content")
        assert filepath.read_text() == "new content"

    def test_list_files_empty_dir(self, tmp_path):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="t")
        client = LLMClient(settings=settings)
        assert client.list_files(str(tmp_path)) == []

    def test_list_files_with_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("y")

        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="t")
        client = LLMClient(settings=settings)
        files = client.list_files(str(tmp_path))
        assert "a.py" in files
        assert str(Path("sub") / "b.py") in files or "sub/b.py" in files

    def test_list_files_nonexistent_dir(self, tmp_path):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="t")
        client = LLMClient(settings=settings)
        assert client.list_files(str(tmp_path / "nonexistent")) == []

    async def test_read_write_roundtrip(self, tmp_path):
        filepath = tmp_path / "roundtrip.txt"
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="t")
        client = LLMClient(settings=settings)

        await client.write_file(str(filepath), "roundtrip content")
        content = await client.read_file(str(filepath))
        assert content == "roundtrip content"
