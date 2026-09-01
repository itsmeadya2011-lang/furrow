from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


class TestLLMClient:
    def test_provider_dispatch_anthropic(self) -> None:
        """Verify anthropic provider is selected."""
        test_settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
        client = LLMClient(settings=test_settings)

        # Mock the _complete_anthropic method
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock, return_value="response") as mock_complete:
            result = asyncio.run(client.complete("test prompt"))

        mock_complete.assert_called_once()
        assert result == "response"

    def test_provider_dispatch_openai(self) -> None:
        """Verify openai provider is selected."""
        test_settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
        client = LLMClient(settings=test_settings)

        # Mock the _complete_openai method
        with patch.object(client, "_complete_openai", new_callable=AsyncMock, return_value="response") as mock_complete:
            result = asyncio.run(client.complete("test prompt"))

        mock_complete.assert_called_once()
        assert result == "response"

    def test_provider_dispatch_ollama(self) -> None:
        """Verify ollama provider is selected (no API key needed)."""
        test_settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=test_settings)

        # Mock the _complete_ollama method
        with patch.object(client, "_complete_ollama", new_callable=AsyncMock, return_value="response") as mock_complete:
            result = asyncio.run(client.complete("test prompt"))

        mock_complete.assert_called_once()
        assert result == "response"

    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path: Path) -> None:
        """Verify read_file returns content."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        test_settings = Settings()
        client = LLMClient(settings=test_settings)

        content = await client.read_file(test_file)
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path: Path) -> None:
        """Verify write_file creates the file."""
        test_file = tmp_path / "subdir" / "test.txt"

        test_settings = Settings()
        client = LLMClient(settings=test_settings)

        await client.write_file(test_file, "new content")

        assert test_file.exists()
        assert test_file.read_text() == "new content"

    def test_list_files(self, tmp_path: Path) -> None:
        """Verify list_files returns relative paths."""
        # Create some files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file2.txt").write_text("content2")
        (tmp_path / "subdir" / "file3.py").write_text("content3")

        test_settings = Settings()
        client = LLMClient(settings=test_settings)

        files = client.list_files(tmp_path)

        assert "file1.txt" in files
        assert str(Path("subdir") / "file2.txt") in files
        assert str(Path("subdir") / "file3.py") in files
