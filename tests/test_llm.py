"""Tests for furrow.llm: LLMClient provider dispatch, file ops, and retry logic."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def make_settings(**kwargs) -> Settings:
    defaults = {
        "provider": Provider.ANTHROPIC,
        "anthropic_api_key": "sk-test-anthropic",
        "openai_api_key": "sk-test-openai",
    }
    defaults.update(kwargs)
    return Settings(**defaults)


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "FURROW_ANTHROPIC_API_KEY", "FURROW_OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return make_settings


async def test_complete_routes_to_anthropic(clean_env):
    client = LLMClient(settings=clean_env())
    client._complete_anthropic = AsyncMock(return_value="anthropic-result")
    client._complete_openai = AsyncMock(return_value="openai-result")
    client._complete_ollama = AsyncMock(return_value="ollama-result")
    result = await client.complete("prompt")
    assert result == "anthropic-result"
    client._complete_anthropic.assert_awaited_once()
    client._complete_openai.assert_not_awaited()
    client._complete_ollama.assert_not_awaited()


async def test_complete_routes_to_openai(clean_env):
    client = LLMClient(settings=clean_env(provider=Provider.OPENAI))
    client._complete_anthropic = AsyncMock(return_value="anthropic-result")
    client._complete_openai = AsyncMock(return_value="openai-result")
    client._complete_ollama = AsyncMock(return_value="ollama-result")
    result = await client.complete("prompt")
    assert result == "openai-result"
    client._complete_openai.assert_awaited_once()
    client._complete_anthropic.assert_not_awaited()
    client._complete_ollama.assert_not_awaited()


async def test_complete_routes_to_ollama(clean_env):
    client = LLMClient(settings=clean_env(provider=Provider.OLLAMA))
    client._complete_anthropic = AsyncMock(return_value="anthropic-result")
    client._complete_openai = AsyncMock(return_value="openai-result")
    client._complete_ollama = AsyncMock(return_value="ollama-result")
    result = await client.complete("prompt")
    assert result == "ollama-result"
    client._complete_ollama.assert_awaited_once()
    client._complete_anthropic.assert_not_awaited()
    client._complete_openai.assert_not_awaited()


async def test_complete_ollama_does_not_raise_value_error(clean_env):
    client = LLMClient(settings=clean_env(provider=Provider.OLLAMA))
    client._complete_ollama = AsyncMock(return_value="ollama-ok")
    result = await client.complete("prompt")
    assert result == "ollama-ok"
    client._complete_ollama.assert_awaited_once()


async def test_complete_retries_on_exception(clean_env, monkeypatch):
    client = LLMClient(settings=clean_env(provider=Provider.ANTHROPIC))
    client.complete.retry.sleep = AsyncMock()
    client._complete_anthropic = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await client.complete("prompt")
    assert client._complete_anthropic.call_count == 3


async def test_write_file_creates_file_and_parents(clean_env, tmp_path):
    client = LLMClient(settings=clean_env())
    target = tmp_path / "sub" / "nested" / "file.txt"
    await client.write_file(target, "hello world")
    assert target.exists()
    assert target.read_text() == "hello world"


async def test_read_file_returns_content(clean_env, tmp_path):
    client = LLMClient(settings=clean_env())
    f = tmp_path / "input.txt"
    f.write_text("file content")
    content = await client.read_file(f)
    assert content == "file content"


async def test_write_then_read_roundtrip(clean_env, tmp_path):
    client = LLMClient(settings=clean_env())
    target = tmp_path / "round.txt"
    await client.write_file(target, "roundtrip-data")
    content = await client.read_file(target)
    assert content == "roundtrip-data"


def test_list_files_discovers_files(clean_env, tmp_path):
    client = LLMClient(settings=clean_env())
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("z")
    files = client.list_files(tmp_path)
    assert set(files) == {"a.py", "b.txt", "sub/c.py"}


def test_list_files_returns_empty_for_missing_dir(clean_env, tmp_path):
    client = LLMClient(settings=clean_env())
    files = client.list_files(tmp_path / "nope")
    assert files == []


def test_anthropic_property_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = client.anthropic


def test_openai_property_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FURROW_OPENAI_API_KEY", raising=False)
    settings = Settings(provider=Provider.OPENAI)
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = client.openai


def test_ollama_property_does_not_require_api_key(clean_env):
    client = LLMClient(settings=clean_env(provider=Provider.OLLAMA))
    result = client.ollama
    assert result is not None
