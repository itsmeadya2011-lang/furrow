from unittest.mock import AsyncMock

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_list_files_existing(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    client = LLMClient(settings=Settings())
    files = client.list_files(tmp_path)
    assert "a.txt" in files
    assert "b.txt" in files


def test_list_files_missing():
    client = LLMClient(settings=Settings())
    assert client.list_files("/nonexistent_path_xyz") == []


async def test_write_and_read_file_roundtrip(tmp_path):
    client = LLMClient(settings=Settings())
    path = tmp_path / "test.txt"
    await client.write_file(path, "hello")
    content = await client.read_file(path)
    assert content == "hello"


async def test_complete_routes_to_anthropic():
    client = LLMClient(settings=Settings())
    client.settings.provider = Provider.ANTHROPIC
    client._complete_anthropic = AsyncMock(return_value="anthropic-out")
    result = await client.complete("prompt")
    assert result == "anthropic-out"


async def test_complete_routes_to_openai():
    client = LLMClient(settings=Settings())
    client.settings.provider = Provider.OPENAI
    client._complete_openai = AsyncMock(return_value="openai-out")
    result = await client.complete("prompt")
    assert result == "openai-out"


async def test_complete_routes_to_ollama():
    client = LLMClient(settings=Settings())
    client.settings.provider = Provider.OLLAMA
    client._complete_ollama = AsyncMock(return_value="ollama-out")
    result = await client.complete("prompt")
    assert result == "ollama-out"


def test_ollama_property_uses_base_url():
    client = LLMClient(settings=Settings())
    client.settings.ollama_base_url = "http://example.com:1234"
    ollama = client.ollama
    assert str(ollama.base_url).endswith("/v1")


def test_anthropic_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(settings=Settings())
    client.settings.anthropic_api_key = None
    assert client.settings.anthropic_api_key is None
    with pytest.raises(ValueError):
        client.anthropic
