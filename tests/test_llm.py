"""Tests for the LLM client wrapper (no network calls anywhere)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from furrow.config import Settings
from furrow.config import settings as default_settings
from furrow.llm import DEFAULT_MAX_TOKENS, LLMClient


class FakeBlock:
    """Stands in for an Anthropic content block."""

    def __init__(self, type: str, text: str = "") -> None:
        self.type = type
        self.text = text


class FakeMessages:
    def __init__(self, content: list[FakeBlock]) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(content=self.content)


class FakeAnthropic:
    def __init__(self, content: list[FakeBlock]) -> None:
        self.messages = FakeMessages(content)


async def test_anthropic_response_picks_the_text_block(settings: Settings, monkeypatch) -> None:
    client = LLMClient(settings=settings)
    fake = FakeAnthropic([FakeBlock("text", "the answer")])
    monkeypatch.setattr(client, "_anthropic", fake)

    assert await client.complete("prompt") == "the answer"
    assert fake.messages.calls[0]["model"] == settings.model


async def test_anthropic_response_skips_non_text_blocks(
    settings: Settings, monkeypatch
) -> None:
    client = LLMClient(settings=settings)
    fake = FakeAnthropic(
        [
            FakeBlock("thinking", "hidden reasoning"),
            FakeBlock("tool_use", "{}"),
            FakeBlock("text", "the answer"),
        ]
    )
    monkeypatch.setattr(client, "_anthropic", fake)

    assert await client.complete("prompt") == "the answer"


async def test_anthropic_response_without_any_text_raises(
    settings: Settings, monkeypatch
) -> None:
    client = LLMClient(settings=settings)
    fake = FakeAnthropic([FakeBlock("thinking", "hidden reasoning")])
    monkeypatch.setattr(client, "_anthropic", fake)

    with pytest.raises(ValueError, match="no text blocks"):
        await client.complete("prompt")


async def test_anthropic_response_joins_multiple_text_blocks(
    settings: Settings, monkeypatch
) -> None:
    client = LLMClient(settings=settings)
    fake = FakeAnthropic([FakeBlock("text", "part one "), FakeBlock("text", "part two")])
    monkeypatch.setattr(client, "_anthropic", fake)

    assert await client.complete("prompt") == "part one part two"


def test_client_without_settings_falls_back_to_the_default_settings() -> None:
    client = LLMClient(settings=None)

    assert client.settings is default_settings
    assert client.settings is not None
    assert client.max_tokens == DEFAULT_MAX_TOKENS


def test_client_uses_the_settings_it_was_given(settings: Settings) -> None:
    client = LLMClient(settings=settings, max_tokens=123)

    assert client.settings is settings
    assert client.max_tokens == 123


def test_list_files_skips_vendor_and_hidden_directories(
    settings: Settings, tmp_path: Path
) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.py").write_text("b", encoding="utf-8")
    for skipped in ("__pycache__", ".git", "node_modules", ".venv"):
        (tmp_path / skipped).mkdir()
        (tmp_path / skipped / "junk.pyc").write_text("x", encoding="utf-8")

    files = LLMClient(settings=settings).list_files(tmp_path)

    assert "a.txt" in files
    assert "pkg/b.py" in files
    assert not any(name.startswith("__pycache__") for name in files)
    assert not any(name.startswith(".git") for name in files)
    assert not any(name.startswith("node_modules") for name in files)
    assert not any(name.startswith(".venv") for name in files)


def test_list_files_on_a_missing_directory_is_empty(settings: Settings, tmp_path: Path) -> None:
    assert LLMClient(settings=settings).list_files(tmp_path / "nope") == []


async def test_write_and_read_file_round_trip(settings: Settings, tmp_path: Path) -> None:
    client = LLMClient(settings=settings)
    target = tmp_path / "nested" / "file.txt"

    await client.write_file(target, "content")

    assert target.read_text(encoding="utf-8") == "content"
    assert await client.read_file(target) == "content"


def test_the_client_never_reaches_for_a_key_on_construction() -> None:
    client = LLMClient(settings=None)

    assert client._anthropic is None
    assert client._openai is None
    assert client._httpx is None
