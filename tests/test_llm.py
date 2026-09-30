from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anthropic
import openai
import pytest

import furrow.llm as llm_mod
from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def _settings(**overrides) -> Settings:
    base: dict = {
        "provider": Provider.OLLAMA,
        "ollama_base_url": "http://test:11434",
    }
    base.update(overrides)
    return Settings(**base)


def _resp(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _make_retryable(exc_type: type) -> BaseException:
    if exc_type is openai.APIError:
        return openai.APIError("nope", request=SimpleNamespace(), body=None)
    return exc_type("nope")


def _install_fake_openai(monkeypatch, behavior):
    state = {"calls": 0}

    class _Completions:
        async def create(self, **kwargs):
            state["calls"] += 1
            return behavior(state["calls"])

    class _Chat:
        @property
        def completions(self) -> _Completions:
            return _Completions()

    class _Client:
        def __init__(self, **kwargs) -> None:
            pass

        @property
        def chat(self) -> _Chat:
            return _Chat()

    monkeypatch.setattr(llm_mod, "AsyncOpenAI", _Client)
    return state


async def test_routing_anthropic(monkeypatch):
    called: dict = {}

    async def fake(self, prompt, system, model):
        called["provider"] = "anthropic"
        return "anthropic-reply"

    monkeypatch.setattr(LLMClient, "_complete_anthropic", fake)
    client = LLMClient(
        settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-test")
    )
    result = await client.complete("hi", model="claude-x")
    assert called["provider"] == "anthropic"
    assert result == "anthropic-reply"


async def test_routing_openai(monkeypatch):
    called: dict = {}

    async def fake(self, prompt, system, model):
        called["provider"] = "openai"
        return "openai-reply"

    monkeypatch.setattr(LLMClient, "_complete_openai", fake)
    client = LLMClient(
        settings=Settings(provider=Provider.OPENAI, openai_api_key="sk-test")
    )
    result = await client.complete("hi", model="gpt-4o")
    assert called["provider"] == "openai"
    assert result == "openai-reply"


async def test_routing_ollama(monkeypatch):
    called: dict = {}

    async def fake(self, prompt, system, model):
        called["provider"] = "ollama"
        return "ollama-reply"

    monkeypatch.setattr(LLMClient, "_complete_ollama", fake)
    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://ollama:11434")
    )
    result = await client.complete("hi", model="llama3")
    assert called["provider"] == "ollama"
    assert result == "ollama-reply"


async def test_ollama_base_url_and_api_key():
    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://custom:9999")
    )
    ollama = client.ollama
    assert ollama.base_url == "http://custom:9999"
    assert ollama.api_key == "ollama"


async def test_ollama_property_cached():
    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://x:11434")
    )
    assert client.ollama is client.ollama


async def test_ollama_default_model_is_llama3(monkeypatch):
    async def fake(self, prompt, system, model):
        return model

    monkeypatch.setattr(LLMClient, "_complete_ollama", fake)
    # settings.model defaults to a claude model, but ollama must default to llama3.
    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://x:11434")
    )
    assert await client.complete("hi") == "llama3"


async def test_ollama_explicit_model_used(monkeypatch):
    async def fake(self, prompt, system, model):
        return model

    monkeypatch.setattr(LLMClient, "_complete_ollama", fake)
    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://x:11434")
    )
    assert await client.complete("hi", model="phi3") == "phi3"


@pytest.mark.parametrize(
    "exc_type",
    [openai.APIError, anthropic.AnthropicError, asyncio.TimeoutError, ConnectionError],
)
async def test_retry_retries_each_retryable_error(monkeypatch, exc_type):
    def behavior(calls):
        if calls < 2:
            raise _make_retryable(exc_type)
        return _resp("recovered")

    state = _install_fake_openai(monkeypatch, behavior)
    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://x:11434")
    )
    result = await client.complete("hi")
    assert result == "recovered"
    assert state["calls"] == 2


async def test_retry_gives_up_after_three(monkeypatch):
    def behavior(calls):
        raise asyncio.TimeoutError("nope")

    state = _install_fake_openai(monkeypatch, behavior)
    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://x:11434")
    )
    with pytest.raises(asyncio.TimeoutError):
        await client.complete("hi")
    assert state["calls"] == 3


async def test_retry_does_not_retry_non_retryable(monkeypatch):
    def behavior(calls):
        raise ValueError("not retryable")

    state = _install_fake_openai(monkeypatch, behavior)
    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://x:11434")
    )
    with pytest.raises(ValueError):
        await client.complete("hi")
    assert state["calls"] == 1
