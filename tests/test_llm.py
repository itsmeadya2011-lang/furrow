"""Tests for LLMClient provider dispatch."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Provider
from furrow.llm import LLMClient


def _make_client(provider) -> LLMClient:
    """Build an LLMClient with a lightweight settings mock.

    This avoids constructing a real Settings/pydantic object and bypasses
    any API-key validation since the _complete_* methods are patched out.
    """
    settings = SimpleNamespace(
        provider=provider,
        model="default-model",
        planner_model="planner-model",
        worker_model="worker-model",
        tester_model="tester-model",
        anthropic_api_key="dummy-anthropic",
        openai_api_key="dummy-openai",
        ollama_base_url="http://localhost:11434",
    )
    return LLMClient(settings=settings)


async def test_complete_routes_anthropic():
    client = _make_client(Provider.ANTHROPIC)
    with patch.object(client, "_complete_anthropic", new_callable=AsyncMock, return_value="ok") as mock:
        result = await client.complete("hi")
        mock.assert_awaited_once_with("hi", "", "default-model")
        assert result == "ok"


async def test_complete_routes_openai():
    client = _make_client(Provider.OPENAI)
    with patch.object(client, "_complete_openai", new_callable=AsyncMock, return_value="ok") as mock:
        result = await client.complete("hi")
        mock.assert_awaited_once_with("hi", "", "default-model")
        assert result == "ok"


async def test_complete_routes_ollama():
    client = _make_client(Provider.OLLAMA)
    with patch.object(client, "_complete_ollama", new_callable=AsyncMock, return_value="ok") as mock:
        result = await client.complete("hi")
        mock.assert_awaited_once_with("hi", "", "default-model")
        assert result == "ok"


async def test_complete_unsupported_provider_raises():
    client = _make_client("bogus")
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hi")


async def test_complete_uses_default_model():
    """When model=None, complete() should fall back to settings.model."""
    client = _make_client(Provider.OLLAMA)
    with patch.object(client, "_complete_ollama", new_callable=AsyncMock, return_value="ok") as mock:
        await client.complete("hi", model=None)
        # complete() resolves model = model or settings.model -> "default-model"
        mock.assert_awaited_once_with("hi", "", "default-model")


async def test_complete_model_override():
    """An explicitly-passed model should be forwarded, not replaced."""
    client = _make_client(Provider.OLLAMA)
    with patch.object(client, "_complete_ollama", new_callable=AsyncMock, return_value="ok") as mock:
        await client.complete("hi", model="custom-model")
        mock.assert_awaited_once_with("hi", "", "custom-model")
