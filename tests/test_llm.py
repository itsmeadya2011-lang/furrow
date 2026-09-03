from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from furrow.config import Provider
from furrow.llm import LLMClient


def test_llm_client_default_provider():
    client = LLMClient()
    assert client.settings.provider == Provider.ANTHROPIC


@pytest.mark.asyncio
async def test_llm_client_routes_to_ollama(monkeypatch):
    client = LLMClient()

    # Fake anthropic/openai clients (should not be called)
    fake_anthropic = MagicMock()
    fake_openai = MagicMock()
    monkeypatch.setattr(client, "_anthropic", fake_anthropic, raising=False)
    monkeypatch.setattr(client, "_openai", fake_openai, raising=False)

    # Fake ollama httpx client
    fake_response = MagicMock()
    fake_response.json.return_value = {"message": {"content": "ollama-out"}}
    fake_response.raise_for_status = MagicMock()
    fake_ollama = MagicMock()
    fake_ollama.post = AsyncMock(return_value=fake_response)
    type(client).ollama = property(lambda self: fake_ollama)

    _complete_anthropic_spy = AsyncMock(return_value="anth-out")
    _complete_openai_spy = AsyncMock(return_value="oai-out")
    _complete_ollama_spy = AsyncMock(return_value="ollama-out")
    monkeypatch.setattr(client, "_complete_anthropic", _complete_anthropic_spy)
    monkeypatch.setattr(client, "_complete_openai", _complete_openai_spy)
    monkeypatch.setattr(client, "_complete_ollama", _complete_ollama_spy)

    monkeypatch.setattr(client.settings, "provider", Provider.OLLAMA)

    result = await client.complete("hi")
    assert result == "ollama-out"
    _complete_ollama_spy.assert_awaited_once()
    _complete_anthropic_spy.assert_not_awaited()
    _complete_openai_spy.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_client_routes_to_anthropic(monkeypatch):
    client = LLMClient()

    _complete_anthropic_spy = AsyncMock(return_value="anth-out")
    _complete_openai_spy = AsyncMock(return_value="oai-out")
    _complete_ollama_spy = AsyncMock(return_value="ollama-out")
    monkeypatch.setattr(client, "_complete_anthropic", _complete_anthropic_spy)
    monkeypatch.setattr(client, "_complete_openai", _complete_openai_spy)
    monkeypatch.setattr(client, "_complete_ollama", _complete_ollama_spy)

    monkeypatch.setattr(client.settings, "provider", Provider.ANTHROPIC)

    result = await client.complete("hi")
    assert result == "anth-out"
    _complete_anthropic_spy.assert_awaited_once()
    _complete_openai_spy.assert_not_awaited()
    _complete_ollama_spy.assert_not_awaited()