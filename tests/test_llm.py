from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


class TestLLMClientInit:
    def test_default_settings(self):
        client = LLMClient()
        assert client.settings is not None
        assert client._anthropic is None
        assert client._openai is None

    def test_custom_settings(self):
        settings = Settings(model="gpt-4", provider=Provider.OPENAI)
        client = LLMClient(settings=settings)
        assert client.settings.model == "gpt-4"
        assert client.settings.provider == Provider.OPENAI


class TestCompleteAnthropic:
    async def test_complete_anthropic(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-ant-test")
        client = LLMClient(settings=settings)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="hello world")]
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        client._anthropic = mock_anthropic
        result = await client.complete("say hi")

        assert result == "hello world"
        mock_anthropic.messages.create.assert_called_once()
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["model"] == settings.model
        assert call_kwargs["max_tokens"] == 4096

    async def test_complete_anthropic_with_system(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-ant-test")
        client = LLMClient(settings=settings)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        client._anthropic = mock_anthropic
        await client.complete("prompt", system="be helpful")

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "be helpful"

    async def test_complete_anthropic_default_system(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-ant-test")
        client = LLMClient(settings=settings)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        client._anthropic = mock_anthropic
        await client.complete("prompt")

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are a helpful coding assistant."


class TestCompleteOpenAI:
    async def test_complete_openai(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key="sk-test")
        client = LLMClient(settings=settings)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="openai response"))]
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        client._openai = mock_openai
        result = await client.complete("say hi")

        assert result == "openai response"

    async def test_complete_openai_empty_content(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key="sk-test")
        client = LLMClient(settings=settings)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=None))]
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        client._openai = mock_openai
        result = await client.complete("say hi")

        assert result == ""


class TestCompleteErrors:
    async def test_unsupported_provider(self):
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("prompt")

    async def test_missing_anthropic_key(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            _ = client.anthropic

    async def test_missing_openai_key(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key=None)
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            _ = client.openai


class TestReadWriteFile:
    async def test_write_and_read_file(self, tmp_path):
        client = LLMClient()
        target = tmp_path / "test.txt"
        await client.write_file(target, "hello world")
        assert target.exists()
        result = await client.read_file(target)
        assert result == "hello world"

    async def test_write_file_creates_directories(self, tmp_path):
        client = LLMClient()
        target = tmp_path / "deep" / "nested" / "file.txt"
        await client.write_file(target, "nested content")
        assert target.exists()
        assert target.read_text() == "nested content"

    async def test_read_missing_file_raises(self, tmp_path):
        client = LLMClient()
        with pytest.raises(FileNotFoundError):
            await client.read_file(tmp_path / "nonexistent.txt")


class TestListFiles:
    def test_list_files_empty_dir(self, tmp_path):
        client = LLMClient()
        result = client.list_files(tmp_path)
        assert result == []

    def test_list_files_with_files(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").write_text("c")

        client = LLMClient()
        result = sorted(client.list_files(tmp_path))
        assert result == ["a.py", "b.py", "sub/c.py"]

    def test_list_files_missing_dir(self, tmp_path):
        client = LLMClient()
        result = client.list_files(tmp_path / "does_not_exist")
        assert result == []
