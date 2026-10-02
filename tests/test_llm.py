"""Tests for the LLMClient."""
from unittest.mock import AsyncMock

import pytest

from furrow.config import Provider, Settings
from furrow.llm import DEFAULT_MAX_ATTEMPTS, DEFAULT_TIMEOUT, RETRYABLE_EXCEPTIONS, LLMClient


class TestLLMClientInit:
    def test_creates_with_default_settings(self):
        client = LLMClient()
        assert client.settings is not None
        assert client._anthropic is None
        assert client._openai is None


class TestCompleteDispatch:
    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        settings = Settings(provider=Provider("invalid"), anthropic_api_key="test")
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("test prompt")

    @pytest.mark.asyncio
    async def test_uses_default_model(self):
        client = LLMClient()
        # Just verify the model resolution logic without making a real call
        model = client.settings.model
        assert model is not None


class TestRetryConstants:
    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 120.0

    def test_default_max_attempts(self):
        assert DEFAULT_MAX_ATTEMPTS == 3

    def test_retryable_exceptions(self):
        import anthropic
        import openai
        assert anthropic.AnthropicError in RETRYABLE_EXCEPTIONS
        assert openai.OpenAIError in RETRYABLE_EXCEPTIONS


class TestCompleteTimeoutParam:
    @pytest.mark.asyncio
    async def test_accepts_timeout_param(self):
        """Verify complete() accepts and passes a timeout parameter."""
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")
        client = LLMClient(settings=settings)
        # Mock the _complete_anthropic method
        client._complete_anthropic = AsyncMock(return_value="response")
        await client.complete("prompt", timeout=60)
        call_kwargs = client._complete_anthropic.call_args
        assert call_kwargs.kwargs["timeout"] == 60


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")
        client = LLMClient(settings=settings)
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        result = await client.read_file(test_file)
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")
        client = LLMClient(settings=settings)
        test_file = tmp_path / "output.txt"
        await client.write_file(test_file, "written content")
        assert test_file.read_text() == "written content"

    @pytest.mark.asyncio
    async def test_write_file_creates_dirs(self, tmp_path):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")
        client = LLMClient(settings=settings)
        test_file = tmp_path / "nested" / "dir" / "output.txt"
        await client.write_file(test_file, "content")
        assert test_file.read_text() == "content"

    def test_list_files(self, tmp_path):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")
        client = LLMClient(settings=settings)
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        files = client.list_files(tmp_path)
        assert "a.txt" in files
        assert "b.txt" in files

    def test_list_files_nonexistent_dir(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")
        client = LLMClient(settings=settings)
        files = client.list_files("/nonexistent/path/12345")
        assert files == []
