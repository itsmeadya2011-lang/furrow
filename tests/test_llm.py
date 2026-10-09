"""Tests for LLMClient - JSON parsing, provider handling, and filesystem helpers."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient, _strip_json_markers


class TestStripJsonMarkers:
    def test_strip_plain_json(self) -> None:
        assert _strip_json_markers('{"key": "value"}').strip() == '{"key": "value"}'

    def test_strip_json_with_language_tag(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        result = _strip_json_markers(text)
        assert result == '{"key": "value"}'

    def test_strip_json_with_generic_fence(self) -> None:
        text = '```\n{"key": "value"}\n```'
        result = _strip_json_markers(text)
        assert result == '{"key": "value"}'

    def test_strip_whitespace(self) -> None:
        assert _strip_json_markers('  {"key": "value"}  ') == '{"key": "value"}'


class TestLLMClient:
    """Tests for LLMClient initialization and provider routing."""

    def test_unsupported_provider_raises(self) -> None:
        settings = Settings()
        # Force a provider that's not handled (simulate via direct call)
        client = LLMClient(settings=settings)
        with patch.object(settings, "provider", "invalid"):
            with pytest.raises(ValueError, match="Unsupported provider"):
                import asyncio

                asyncio.run(client.complete("test"))

    @pytest.mark.asyncio
    async def test_complete_json_success(self) -> None:
        settings = Settings()
        client = LLMClient(settings=settings)
        client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok"}')
        result = await client.complete_json("test")
        assert result == {"passed": True, "summary": "ok"}

    @pytest.mark.asyncio
    async def test_complete_json_strips_markers(self) -> None:
        settings = Settings()
        client = LLMClient(settings=settings)
        client.complete = AsyncMock(return_value='```json\n{"key": "value"}\n```')
        result = await client.complete_json("test")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_complete_json_parse_error(self) -> None:
        settings = Settings()
        client = LLMClient(settings=settings)
        client.complete = AsyncMock(return_value="not json at all")
        with pytest.raises(json.JSONDecodeError):
            await client.complete_json("test")

    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        client = LLMClient(settings=Settings())
        content = await client.read_file(f)
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_write_file_creates_dirs(self, tmp_path: Path) -> None:
        settings = Settings(workspace=tmp_path)
        client = LLMClient(settings=settings)
        target = tmp_path / "subdir" / "file.txt"
        await client.write_file(target, "content here")
        assert target.exists()
        assert target.read_text() == "content here"

    def test_list_files(self, tmp_path: Path) -> None:
        settings = Settings(workspace=tmp_path)
        client = LLMClient(settings=settings)
        (tmp_path / "a.py").touch()
        (tmp_path / "b.txt").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").touch()
        files = client.list_files(tmp_path)
        assert "a.py" in files
        assert "b.txt" in files
        assert "sub/c.py" in files

    def test_ollama_client_init(self) -> None:
        """Ollama client should init with the configured base URL."""
        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)
        # Should not raise
        ollama = client.ollama
        assert ollama.base_url == "http://localhost:11434"
