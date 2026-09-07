import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.mark.asyncio
async def test_complete_anthropic():
    settings = Settings(provider=Provider.ANTHROPIC)
    llm = LLMClient(settings=settings)
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="anthropic result")]
    llm._anthropic = MagicMock()
    llm._anthropic.messages.create = AsyncMock(return_value=mock_response)

    result = await llm.complete("prompt", model="claude-test")
    assert result == "anthropic result"
    llm._anthropic.messages.create.assert_called_once()


@pytest.mark.asyncio
async def test_complete_openai():
    settings = Settings(provider=Provider.OPENAI)
    llm = LLMClient(settings=settings)
    mock_message = MagicMock()
    mock_message.content = "openai result"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    llm._openai = MagicMock()
    llm._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await llm.complete("prompt", model="gpt-test")
    assert result == "openai result"
    llm._openai.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_complete_ollama_uses_httpx():
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    llm = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "ollama result"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.post = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)

        result = await llm.complete("prompt", model="llama-test")

    assert result == "ollama result"
    instance.post.assert_called_once()
    call_args = instance.post.call_args
    assert call_args.args[0] == "http://localhost:11434/api/generate"
    payload = call_args.kwargs["json"]
    assert payload["model"] == "llama-test"
    assert payload["prompt"] == "prompt"
    assert payload["stream"] is False


@pytest.mark.asyncio
async def test_read_file(tmp_path):
    settings = Settings()
    llm = LLMClient(settings=settings)
    target = tmp_path / "data.txt"
    target.write_text("hello world")

    result = await llm.read_file(str(target))
    assert result == "hello world"


@pytest.mark.asyncio
async def test_write_file(tmp_path):
    settings = Settings(workspace=tmp_path)
    llm = LLMClient(settings=settings)

    target = tmp_path / "subdir" / "file.txt"
    await llm.write_file(str(target), "content here")

    assert target.exists()
    assert target.read_text() == "content here"


def test_list_files_empty_when_missing():
    settings = Settings()
    llm = LLMClient(settings=settings)
    result = llm.list_files("/nonexistent/path/12345")
    assert result == []
