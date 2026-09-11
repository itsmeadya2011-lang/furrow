from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_llm_client_uses_anthropic_provider() -> None:
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient(settings=settings)
    assert client.settings.provider == Provider.ANTHROPIC


def test_llm_client_uses_openai_provider() -> None:
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)
    assert client.settings.provider == Provider.OPENAI


def test_llm_client_uses_ollama_provider() -> None:
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)
    assert client.settings.provider == Provider.OLLAMA


@pytest.mark.asyncio
async def test_llm_client_complete_ollama() -> None:
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "hello from ollama"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await client.complete("hi", model="llama3")
        assert result == "hello from ollama"


@pytest.mark.asyncio
async def test_llm_client_write_file_creates_dirs(tmp_path: Path) -> None:
    settings = Settings()
    client = LLMClient(settings=settings)
    path = tmp_path / "subdir" / "file.txt"
    await client.write_file(path, "content")
    assert path.read_text() == "content"


@pytest.mark.asyncio
async def test_llm_client_read_file(tmp_path: Path) -> None:
    settings = Settings()
    client = LLMClient(settings=settings)
    path = tmp_path / "file.txt"
    path.write_text("hello")
    result = await client.read_file(path)
    assert result == "hello"


def test_llm_client_list_files(tmp_path: Path) -> None:
    settings = Settings()
    client = LLMClient(settings=settings)
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "b.py").write_text("y")
    files = client.list_files(tmp_path)
    assert "a.py" in files
    assert any("b.py" in f for f in files)