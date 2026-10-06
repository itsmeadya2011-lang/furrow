from __future__ import annotations

import asyncio

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def _settings(provider=Provider.ANTHROPIC, **kw):
    return Settings(provider=provider, **kw)


# --- provider routing ---

def test_complete_routes_to_anthropic(monkeypatch):
    client = LLMClient(settings=_settings(Provider.ANTHROPIC, anthropic_api_key="sk-x"))
    called = {}

    async def fake(prompt, system, model):
        called["model"] = model
        return "anthropic-answer"

    monkeypatch.setattr(client, "_complete_anthropic", fake)
    out = asyncio.run(client.complete("hi", system="s", model="m"))
    assert out == "anthropic-answer"
    assert called["model"] == "m"


def test_complete_routes_openai_with_openai_client(monkeypatch):
    client = LLMClient(settings=_settings(Provider.OPENAI, openai_api_key="sk-o"))
    called = {}

    async def fake(prompt, system, model, client_):
        called["passed_client"] = client_
        return "openai-answer"

    monkeypatch.setattr(client, "_complete_openai", fake)
    out = asyncio.run(client.complete("hi"))
    assert out == "openai-answer"
    assert called["passed_client"] is client.openai


def test_ollama_uses_ollama_client(monkeypatch):
    client = LLMClient(settings=_settings(Provider.OLLAMA))
    called = {}

    async def fake(prompt, system, model, client_):
        called["passed_client"] = client_
        return "ollama-answer"

    monkeypatch.setattr(client, "_complete_openai", fake)
    out = asyncio.run(client.complete("hi"))
    assert out == "ollama-answer"
    assert called["passed_client"] is client.ollama
    assert str(client.ollama.base_url).rstrip("/") == client.settings.ollama_base_url + "/v1"


def test_unknown_provider_raises(monkeypatch):
    client = LLMClient(settings=_settings(Provider.ANTHROPIC, anthropic_api_key="sk-x"))
    client.settings.provider = "unknown-provider"
    with pytest.raises(ValueError, match="Unsupported provider"):
        asyncio.run(client.complete("hi"))


# --- retry behavior ---

def test_complete_retries_then_reraises(monkeypatch):
    client = LLMClient(settings=_settings(Provider.ANTHROPIC, anthropic_api_key="sk-x"))
    calls = {"n": 0}

    async def boom(prompt, system, model):
        calls["n"] += 1
        raise ConnectionError("transient")

    monkeypatch.setattr(client, "_complete_anthropic", boom)

    async def fast_sleep(*args, **kwargs):
        return 0

    monkeypatch.setattr("asyncio.sleep", fast_sleep)
    with pytest.raises(ConnectionError):
        asyncio.run(client.complete("hi"))
    assert calls["n"] == 3
