from __future__ import annotations

import pytest

from furrow.config import Provider
from furrow.llm import LLMClient
from tests.conftest import make_settings


async def test_write_file_creates_parent_dirs(tmp_path):
    client = LLMClient(settings=make_settings())
    target = tmp_path / "sub" / "deep" / "f.txt"
    await client.write_file(target, "hello")
    assert target.read_text() == "hello"


async def test_read_file_returns_written_content(tmp_path):
    client = LLMClient(settings=make_settings())
    await client.write_file(tmp_path / "a.txt", "contents here")
    assert await client.read_file(tmp_path / "a.txt") == "contents here"


async def test_list_files_returns_relative_paths(tmp_path):
    client = LLMClient(settings=make_settings())
    await client.write_file(tmp_path / "a.txt", "1")
    await client.write_file(tmp_path / "b" / "c.txt", "2")
    files = client.list_files(tmp_path)
    assert "a.txt" in files
    assert any(f.endswith("c.txt") for f in files)


async def test_list_files_missing_dir_returns_empty(tmp_path):
    client = LLMClient(settings=make_settings())
    assert client.list_files(tmp_path / "does-not-exist") == []


async def test_complete_dispatches_to_ollama(monkeypatch):
    client = LLMClient(settings=make_settings(provider=Provider.OLLAMA))

    async def fake(prompt, system, model):
        return "ok"

    monkeypatch.setattr(client, "_complete_ollama", fake)
    assert await client.complete("hi") == "ok"


async def test_complete_dispatches_to_anthropic(monkeypatch):
    client = LLMClient(settings=make_settings(provider=Provider.ANTHROPIC))

    async def fake(prompt, system, model):
        return "ok"

    monkeypatch.setattr(client, "_complete_anthropic", fake)
    assert await client.complete("hi") == "ok"


async def test_complete_retries_then_succeeds(monkeypatch):
    client = LLMClient(settings=make_settings(provider=Provider.OLLAMA))
    calls = {"n": 0}

    async def flaky(prompt, system, model):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return "recovered"

    monkeypatch.setattr(client, "_complete_ollama", flaky)
    result = await client.complete("hi")
    assert result == "recovered"
    assert calls["n"] == 2


async def test_complete_unsupported_provider():
    settings = make_settings()
    settings.provider = "bogus"  # type: ignore[assignment]
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hi")
