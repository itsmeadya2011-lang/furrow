from unittest.mock import AsyncMock

import pytest

from furrow.llm import LLMClient
from furrow.config import Provider, Settings


class _FakeResponse:
    def json(self):
        return {"message": {"content": "ollama-out"}}

    def raise_for_status(self):
        pass


class _FakeOllamaClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def post(self, *args, **kwargs):
        return _FakeResponse()


async def test_llm_client_anthropic(monkeypatch):
    fake_anthropic = AsyncMock()
    fake_anthropic.messages.create = AsyncMock(
        return_value=type("Resp", (), {"content": [type("B", (), {"text": "hello"})()]})()
    )
    monkeypatch.setattr(LLMClient, "anthropic", property(lambda self: fake_anthropic))

    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    result = await client.complete("hi")
    assert result == "hello"
    fake_anthropic.messages.create.assert_called_once()


async def test_llm_client_openai(monkeypatch):
    fake_openai = AsyncMock()
    fake_openai.chat.completions.create = AsyncMock(
        return_value=type(
            "Resp",
            (),
            {
                "choices": [
                    type("C", (), {"message": type("M", (), {"content": "hi"})()})()
                ]
            },
        )()
    )
    monkeypatch.setattr(LLMClient, "openai", property(lambda self: fake_openai))

    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    result = await client.complete("hi")
    assert result == "hi"
    fake_openai.chat.completions.create.assert_called_once()


async def test_llm_client_ollama(monkeypatch):
    monkeypatch.setattr("httpx.AsyncClient", _FakeOllamaClient)

    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    )
    result = await client.complete("hi")
    assert result == "ollama-out"
