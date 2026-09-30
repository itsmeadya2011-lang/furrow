from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.mark.asyncio
async def test_complete_dispatches_to_ollama(monkeypatch):
    client = LLMClient(Settings(provider=Provider.OLLAMA))
    mock_complete = AsyncMock(return_value="ok")
    monkeypatch.setattr(client, "_complete_ollama", mock_complete)

    result = await client.complete("hi")

    assert result == "ok"
    mock_complete.assert_awaited_once_with("hi", "", client.settings.model)


@pytest.mark.asyncio
async def test_complete_ollama_request_construction(monkeypatch):
    client = LLMClient(
        Settings(provider=Provider.OLLAMA, ollama_base_url="http://test")
    )

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"message": {"content": "hi from ollama"}}

    ollama_mock = AsyncMock()
    ollama_mock.post = AsyncMock(return_value=response)
    monkeypatch.setattr(LLMClient, "ollama", ollama_mock)

    result = await client._complete_ollama("prompt", "sys", "model")

    assert result == "hi from ollama"
    ollama_mock.post.assert_awaited_once()
    call = ollama_mock.post.call_args
    assert call.args[0] == "http://test/api/chat"
    payload = call.kwargs["json"]
    assert payload == {
        "model": "model",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prompt"},
        ],
        "stream": False,
    }


@pytest.mark.asyncio
async def test_complete_ollama_defaults(monkeypatch):
    client = LLMClient(
        Settings(provider=Provider.OLLAMA, ollama_base_url="http://test")
    )

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"message": {"content": "default"}}

    ollama_mock = AsyncMock()
    ollama_mock.post = AsyncMock(return_value=response)
    monkeypatch.setattr(LLMClient, "ollama", ollama_mock)

    result = await client._complete_ollama("hello", "", None)

    assert result == "default"
    call = ollama_mock.post.call_args
    payload = call.kwargs["json"]
    assert payload["model"] == "llama3"
    assert payload["messages"][0]["content"] == "You are a helpful coding assistant."


@pytest.mark.asyncio
async def test_ollama_property_binds_base_url():
    client = LLMClient(
        Settings(provider=Provider.OLLAMA, ollama_base_url="http://test")
    )

    assert isinstance(client.ollama, httpx.AsyncClient)
    assert "http://test" in str(client.ollama.base_url)
    await client.ollama.aclose()
