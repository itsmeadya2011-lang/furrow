import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_complete_unsupported_provider_raises():
    settings = SimpleNamespace(
        provider="bogus",
        model="m",
        worker_model="m",
        planner_model="m",
        tester_model="m",
    )
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="Unsupported provider"):
        asyncio.run(client.complete("hello"))


async def test_complete_missing_anthropic_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        await client.complete("hello")


async def test_write_and_read_file(tmp_path):
    client = LLMClient()
    target = tmp_path / "sub" / "note.txt"
    await client.write_file(target, "hello")
    assert target.read_text() == "hello"
    content = await client.read_file(target)
    assert content == "hello"


def test_list_files(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    client = LLMClient()
    files = sorted(client.list_files(tmp_path))
    assert files == ["a.txt", "sub/b.txt"]

    assert client.list_files(tmp_path / "missing") == []
