from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.llm import LLMClient
from furrow.config import Provider, Settings


def _settings(provider: Provider = Provider.ANTHROPIC, **kwargs) -> Settings:
    data = {"provider": provider, **kwargs}
    return Settings(**data)


def test_llm_client_defaults():
    client = LLMClient()
    assert client.settings.provider == Provider.ANTHROPIC


def test_llm_client_custom_settings():
    settings = _settings(provider=Provider.OPENAI, openai_api_key="sk-test")
    client = LLMClient(settings=settings)
    assert client.settings.provider == Provider.OPENAI


@pytest.mark.asyncio
async def test_complete_anthropic():
    settings = _settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-ant-test")
    client = LLMClient(settings=settings)
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="hello world")]
    with patch.object(client, "_anthropic", None):
        with patch("furrow.llm.AsyncAnthropic") as mock_cls:
            instance = AsyncMock()
            instance.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = instance
            result = await client.complete("say hi")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_complete_openai():
    settings = _settings(provider=Provider.OPENAI, openai_api_key="sk-test")
    client = LLMClient(settings=settings)
    mock_choice = MagicMock()
    mock_choice.message.content = "hello openai"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    with patch.object(client, "_openai", None):
        with patch("furrow.llm.AsyncOpenAI") as mock_cls:
            instance = AsyncMock()
            instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = instance
            result = await client.complete("say hi")
    assert result == "hello openai"


@pytest.mark.asyncio
async def test_complete_ollama():
    settings = _settings(
        provider=Provider.OLLAMA,
        ollama_base_url="http://localhost:11434",
    )
    client = LLMClient(settings=settings)
    mock_choice = MagicMock()
    mock_choice.message.content = "hello ollama"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    with patch.object(client, "_openai_ollama", None):
        with patch("furrow.llm.AsyncOpenAI") as mock_cls:
            instance = AsyncMock()
            instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = instance
            result = await client.complete("say hi")
    assert result == "hello ollama"


def test_complete_unsupported_provider():
    settings = _settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)
    with patch.object(client.settings, "provider", "unsupported"):
        with pytest.raises(ValueError, match="Unsupported provider"):
            # We need to bypass enum validation for this test
            client.settings.provider = "unsupported"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_read_write_file(tmp_path: Path):
    settings = _settings()
    client = LLMClient(settings=settings)
    target = tmp_path / "hello.txt"
    await client.write_file(target, "hello")
    assert target.read_text() == "hello"
    content = await client.read_file(target)
    assert content == "hello"


@pytest.mark.asyncio
async def test_write_file_creates_dirs(tmp_path: Path):
    settings = _settings()
    client = LLMClient(settings=settings)
    target = tmp_path / "a" / "b" / "hello.txt"
    await client.write_file(target, "hello")
    assert target.read_text() == "hello"


def test_list_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")

    settings = _settings()
    client = LLMClient(settings=settings)
    files = sorted(client.list_files(tmp_path))
    assert files == ["a.txt", "b.txt", "sub/c.txt"]


def test_list_files_empty():
    settings = _settings()
    client = LLMClient(settings=settings)
    assert client.list_files("/nonexistent/path/12345") == []
