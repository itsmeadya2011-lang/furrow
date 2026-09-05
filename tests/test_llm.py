import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Settings, settings
from furrow.llm import LLMClient


def test_llm_client_defaults():
    client = LLMClient()
    assert client.settings == settings


def test_llm_client_missing_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    client = LLMClient()
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = client.anthropic


def test_llm_client_missing_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None)
    client = LLMClient()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = client.openai


def test_llm_client_ollama_property():
    client = LLMClient()
    ollama = client.ollama
    assert ollama is client.ollama  # cached
    assert ollama.base_url == settings.ollama_base_url


@pytest.mark.asyncio
async def test_llm_client_complete_routes_anthropic(monkeypatch):
    client = LLMClient()
    mock_response = AsyncMock()
    mock_response.content[0].text = "hello"
    monkeypatch.setattr(client.anthropic.messages, "create", AsyncMock(return_value=mock_response))
    result = await client.complete("prompt", model="claude-test")
    assert result == "hello"


@pytest.mark.asyncio
async def test_llm_client_complete_routes_openai(monkeypatch):
    monkeypatch.setattr(settings, "provider", "openai")
    client = LLMClient()
    mock_choice = AsyncMock()
    mock_choice.message.content = "hello"
    mock_response = AsyncMock()
    mock_response.choices = [mock_choice]
    monkeypatch.setattr(client.openai.chat.completions, "create", AsyncMock(return_value=mock_response))
    result = await client.complete("prompt", model="gpt-test")
    assert result == "hello"


def test_list_files_nonexistent():
    client = LLMClient()
    assert client.list_files("/nonexistent/path/12345") == []


def test_list_files_existing(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    client = LLMClient()
    files = client.list_files(tmp_path)
    assert set(files) == {"a.txt", "b.txt"}


@pytest.mark.asyncio
async def test_read_write_file(tmp_path):
    client = LLMClient()
    target = tmp_path / "test.txt"
    await client.write_file(target, "hello world")
    assert target.read_text() == "hello world"
    content = await client.read_file(target)
    assert content == "hello world"
