from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_list_files(tmp_path):
    (tmp_path / "a.py").write_text("print('a')")
    (tmp_path / "b.py").write_text("print('b')")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("print('c')")

    client = LLMClient()
    files = client.list_files(tmp_path)
    assert "a.py" in files
    assert "b.py" in files
    assert "sub/c.py" in files
    assert len(files) == 3


def test_list_files_empty_dir(tmp_path):
    client = LLMClient()
    files = client.list_files(tmp_path)
    assert files == []


def test_list_files_nonexistent():
    client = LLMClient()
    files = client.list_files("/nonexistent/path/xyz")
    assert files == []


def test_read_and_write_file(tmp_path):
    client = LLMClient()
    path = tmp_path / "test.txt"
    import asyncio
    asyncio.run(client.write_file(path, "hello world"))
    assert path.exists()
    content = asyncio.run(client.read_file(path))
    assert content == "hello world"


def test_write_file_creates_parents(tmp_path):
    client = LLMClient()
    path = tmp_path / "nested" / "dir" / "file.txt"
    import asyncio
    asyncio.run(client.write_file(path, "content"))
    assert path.exists()
    assert path.read_text() == "content"


def test_provider_anthropic_requires_key():
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = client.anthropic


def test_provider_openai_requires_key():
    settings = Settings(provider=Provider.OPENAI, openai_api_key=None)
    os.environ.pop("OPENAI_API_KEY", None)
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = client.openai


def test_provider_ollama():
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    # Ollama doesn't need API key, should not raise
    assert client.settings.provider == Provider.OLLAMA