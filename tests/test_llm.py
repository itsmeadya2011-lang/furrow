"""Tests for the LLM client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Provider, Settings
from furrow.llm import LLMClient, LLMError


@pytest.fixture
def mock_settings():
    return Settings(
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        anthropic_api_key="test-key",
    )


@pytest.fixture
def client(mock_settings):
    return LLMClient(settings=mock_settings)


class TestLLMClient:
    def test_init_with_settings(self, mock_settings):
        client = LLMClient(settings=mock_settings)
        assert client.settings == mock_settings

    def test_init_default_settings(self):
        client = LLMClient()
        assert client.settings is not None

    @pytest.mark.asyncio
    async def test_complete_anthropic_success(self, client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]

        with patch("furrow.llm.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_instance

            result = await client.complete("Test prompt")
            assert result == "Test response"

    @pytest.mark.asyncio
    async def test_complete_openai_success(self, mock_settings):
        mock_settings.provider = Provider.OPENAI
        mock_settings.openai_api_key = "test-key"
        client = LLMClient(settings=mock_settings)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OpenAI response"))]

        with patch("furrow.llm.AsyncOpenAI") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_instance

            result = await client.complete("Test prompt")
            assert result == "OpenAI response"

    @pytest.mark.asyncio
    async def test_complete_unsupported_provider(self, mock_settings):
        mock_settings.provider = "invalid"
        client = LLMClient(settings=mock_settings)

        with pytest.raises(LLMError, match="Unsupported provider"):
            await client.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_uses_custom_model(self, client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]

        with patch("furrow.llm.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_instance

            await client.complete("Test", model="custom-model")
            mock_instance.messages.create.assert_called_once()
            call_kwargs = mock_instance.messages.create.call_args[1]
            assert call_kwargs["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_read_file(self, client, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = await client.read_file(test_file)
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_write_file(self, client, tmp_path):
        test_file = tmp_path / "subdir" / "test.txt"
        await client.write_file(test_file, "Test content")

        assert test_file.exists()
        assert test_file.read_text() == "Test content"

    def test_list_files(self, client, tmp_path):
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.py").write_text("content2")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.txt").write_text("content3")

        files = client.list_files(tmp_path)
        assert "file1.txt" in files
        assert "file2.py" in files
        assert "subdir/file3.txt" in files

    def test_list_files_nonexistent(self, client):
        files = client.list_files("/nonexistent/path")
        assert files == []