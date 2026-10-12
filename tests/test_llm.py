from __future__ import annotations

from pathlib import Path

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_settings_defaults() -> None:
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.workspace == Path.cwd()


def test_settings_env_prefix() -> None:
    import os

    os.environ["FURROW_PROVIDER"] = "openai"
    try:
        s = Settings()
        assert s.provider == Provider.OPENAI
    finally:
        del os.environ["FURROW_PROVIDER"]


def test_settings_ollama_provider() -> None:
    import os

    os.environ["FURROW_PROVIDER"] = "ollama"
    try:
        s = Settings()
        assert s.provider == Provider.OLLAMA
        assert s.ollama_base_url == "http://localhost:11434"
    finally:
        del os.environ["FURROW_PROVIDER"]


def test_settings_max_cycles_from_env() -> None:
    import os

    os.environ["FURROW_MAX_CYCLES"] = "3"
    try:
        s = Settings()
        assert s.max_cycles == 3
    finally:
        del os.environ["FURROW_MAX_CYCLES"]


def test_settings_max_parallel_from_env() -> None:
    import os

    os.environ["FURROW_MAX_PARALLEL_TASKS"] = "8"
    try:
        s = Settings()
        assert s.max_parallel_tasks == 8
    finally:
        del os.environ["FURROW_MAX_PARALLEL_TASKS"]


@pytest.mark.asyncio
async def test_complete_unsupported_provider_raises() -> None:
    settings = Settings()
    settings.provider = "bogus"  # type: ignore[assignment]
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hello")


@pytest.mark.asyncio
async def test_anthropic_missing_key_raises() -> None:
    import os

    os.environ.pop("ANTHROPIC_API_KEY", None)
    settings = Settings()
    settings.anthropic_api_key = None
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = client.anthropic


@pytest.mark.asyncio
async def test_openai_missing_key_raises() -> None:
    import os

    os.environ.pop("OPENAI_API_KEY", None)
    settings = Settings()
    settings.openai_api_key = None
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = client.openai


@pytest.mark.asyncio
async def test_write_and_read_file(tmp_path: Path) -> None:
    client = LLMClient()
    target = tmp_path / "out.txt"
    await client.write_file(target, "hello world")
    assert target.read_text() == "hello world"
    assert await client.read_file(target) == "hello world"


def test_list_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.md").write_text("# hi")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("y")

    client = LLMClient()
    files = client.list_files(tmp_path)
    assert "a.py" in files
    assert "b.md" in files
    assert any("sub" in f and "c.txt" in f for f in files)


def test_list_files_missing_dir_returns_empty() -> None:
    client = LLMClient()
    assert client.list_files("/nonexistent/path/xyz") == []