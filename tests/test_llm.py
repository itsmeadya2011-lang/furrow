from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Settings
from furrow.llm import LLMClient


class TestLLMClient:
    def test_missing_anthropic_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("furrow.llm.settings") as mock_settings:
            mock_settings.anthropic_api_key = None
            mock_settings.provider = "anthropic"
            client = LLMClient(settings=mock_settings)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                _ = client.anthropic

    def test_missing_openai_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("furrow.llm.settings") as mock_settings:
            mock_settings.openai_api_key = None
            mock_settings.provider = "openai"
            client = LLMClient(settings=mock_settings)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                _ = client.openai

    def test_unsupported_provider_raises(self):
        with patch("furrow.llm.settings") as mock_settings:
            mock_settings.provider = "unknown"
            client = LLMClient(settings=mock_settings)
            with pytest.raises(ValueError, match="Unsupported provider"):
                import asyncio
                asyncio.run(client.complete("test"))

    def test_list_files_empty_for_missing_dir(self):
        client = LLMClient()
        result = client.list_files("/nonexistent/path/12345")
        assert result == []

    def test_list_files_returns_relative_paths(self, tmp_path: Path):
        client = LLMClient()
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "file.txt").write_text("hello")
        result = client.list_files(str(tmp_path))
        assert "sub/file.txt" in result
