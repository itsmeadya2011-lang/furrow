"""Tests for the LLMClient."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def _make_settings(provider: Provider = Provider.ANTHROPIC) -> Settings:
    """Create a Settings instance with test-friendly defaults."""
    # Use a temp env to avoid needing real API keys
    with patch.dict(os.environ, {"FURROW_ANTHROPIC_API_KEY": "test-key"}):
        return Settings(
            provider=provider,
            max_tokens=1024,
            timeout=5.0,
            max_retries=1,
            retry_delay=0.01,
        )


class TestLLMClientInit:
    def test_default_settings(self):
        client = LLMClient()
        assert client.settings is not None
        assert client._anthropic is None
        assert client._openai is None

    def test_custom_settings(self):
        custom = _make_settings()
        client = LLMClient(settings=custom)
        assert client.settings is custom


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_complete_anthropic_returns_text(self):
        settings = _make_settings(Provider.ANTHROPIC)
        client = LLMClient(settings=settings)
        # Mock the anthropic property
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from Claude")]
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        client._anthropic = mock_anthropic

        result = await client.complete("test prompt", system="sys", model="claude-test")
        assert result == "Hello from Claude"

    @pytest.mark.asyncio
    async def test_complete_uses_custom_system(self):
        settings = _make_settings(Provider.ANTHROPIC)
        client = LLMClient(settings=settings)
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        client._anthropic = mock_anthropic

        await client.complete("prompt", system="custom system", model="claude-test")
        call_kwargs = mock_anthropic.messages.create.call_args
        assert call_kwargs.kwargs["system"] == "custom system"

    @pytest.mark.asyncio
    async def test_complete_uses_default_system(self):
        settings = _make_settings(Provider.ANTHROPIC)
        client = LLMClient(settings=settings)
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        client._anthropic = mock_anthropic

        await client.complete("prompt", system="", model="claude-test")
        call_kwargs = mock_anthropic.messages.create.call_args
        assert "helpful coding assistant" in call_kwargs.kwargs["system"]


class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_complete_openai_returns_text(self):
        settings = _make_settings(Provider.OPENAI)
        with patch.dict(os.environ, {"FURROW_OPENAI_API_KEY": "test-key"}):
            client = LLMClient(settings=settings)
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenAI"))]
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        client._openai = mock_openai

        result = await client.complete("test prompt", system="sys", model="gpt-test")
        assert result == "Hello from OpenAI"


class TestUnsupportedProvider:
    @pytest.mark.asyncio
    async def test_complete_unsupported_provider_raises(self):
        settings = _make_settings(Provider.OLLAMA)
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("test")


class TestFileOperations:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        settings = _make_settings(Provider.ANTHROPIC)
        client = LLMClient(settings=settings)
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        result = await client.read_file(test_file)
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_write_file_creates_dirs(self, tmp_path):
        settings = _make_settings(Provider.ANTHROPIC)
        client = LLMClient(settings=settings)
        # Override settings.workspace for resolve_path test
        client.settings.workspace = tmp_path

        target = tmp_path / "subdir" / "output.txt"
        await client.write_file(target, "content here")
        assert target.read_text() == "content here"

    def test_list_files(self, tmp_path):
        settings = _make_settings(Provider.ANTHROPIC)
        client = LLMClient(settings=settings)

        (tmp_path / "a.py").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("y")

        result = client.list_files(tmp_path)
        assert "a.py" in result
        assert "sub/b.py" in result

    def test_list_files_nonexistent(self):
        settings = _make_settings(Provider.ANTHROPIC)
        client = LLMClient(settings=settings)
        assert client.list_files("/nonexistent/path/12345") == []


class TestResolvePath:
    def test_resolve_relative(self):
        from pathlib import Path
        settings = _make_settings(Provider.ANTHROPIC)
        settings.workspace = Path("/test/workspace")
        client = LLMClient(settings=settings)
        result = client.resolve_path("src/main.py")
        assert result == Path("/test/workspace/src/main.py")

    def test_resolve_absolute(self):
        from pathlib import Path
        settings = _make_settings(Provider.ANTHROPIC)
        client = LLMClient(settings=settings)
        result = client.resolve_path("/absolute/path.py")
        assert result == Path("/absolute/path.py")
