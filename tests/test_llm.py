from __future__ import annotations

import json
from pathlib import Path

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


class _DummySettings(Settings):
    """Settings that avoid reading .env and default to a safe provider."""

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)


def test_complete_json_returns_valid_json(monkeypatch):
    settings = _DummySettings(provider=Provider.ANTHROPIC, anthropic_api_key="x")
    client = LLMClient(settings=settings)

    async def fake_complete(prompt, system="", model=None):
        return '{"ok": true}'

    monkeypatch.setattr(client, "complete", fake_complete)
    result = __import__("asyncio").run(client.complete_json("do thing"))
    assert json.loads(result) == {"ok": True}


def test_complete_json_retries_then_succeeds(monkeypatch):
    settings = _DummySettings(provider=Provider.ANTHROPIC, anthropic_api_key="x")
    client = LLMClient(settings=settings)
    calls = {"n": 0}

    async def fake_complete(prompt, system="", model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json"
        return '{"ok": true}'

    monkeypatch.setattr(client, "complete", fake_complete)
    result = __import__("asyncio").run(client.complete_json("do thing"))
    assert json.loads(result) == {"ok": True}
    assert calls["n"] == 2


def test_complete_json_raises_after_two_failures(monkeypatch):
    settings = _DummySettings(provider=Provider.ANTHROPIC, anthropic_api_key="x")
    client = LLMClient(settings=settings)

    async def fake_complete(prompt, system="", model=None):
        return "still not json"

    monkeypatch.setattr(client, "complete", fake_complete)
    with pytest.raises(ValueError):
        __import__("asyncio").run(client.complete_json("do thing"))


def test_ollama_provider_uses_httpx(monkeypatch):
    settings = _DummySettings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "hello from ollama"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def post(self, url, json=None):
            captured["url"] = url
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr("furrow.llm.httpx.AsyncClient", FakeAsyncClient)
    # Force lazy client creation
    client._httpx = FakeAsyncClient()  # type: ignore[assignment]
    result = __import__("asyncio").run(client.complete("hi", system="sys", model="llama3"))
    assert result == "hello from ollama"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["payload"]["model"] == "llama3"
    assert captured["payload"]["stream"] is False


def test_read_and_write_file(tmp_path: Path):
    settings = _DummySettings(provider=Provider.ANTHROPIC, anthropic_api_key="x")
    client = LLMClient(settings=settings)
    target = tmp_path / "sub" / "file.txt"

    __import__("asyncio").run(client.write_file(target, "hello world"))
    assert target.read_text() == "hello world"

    result = __import__("asyncio").run(client.read_file(target))
    assert result == "hello world"


def test_list_files_returns_relative_paths(tmp_path: Path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.py").write_text("y")

    settings = _DummySettings(provider=Provider.ANTHROPIC, anthropic_api_key="x", workspace=tmp_path)
    client = LLMClient(settings=settings)
    files = client.list_files(tmp_path)
    assert "a.py" in files
    assert str(Path("nested") / "b.py") in files