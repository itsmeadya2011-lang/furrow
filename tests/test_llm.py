from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


async def test_provider_routing_anthropic() -> None:
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    with patch.object(
        client, "_complete_anthropic", new_callable=AsyncMock, return_value="anthropic result"
    ) as mock_method:
        result = await client.complete("hi", system="sys")

    mock_method.assert_awaited_once()
    assert result == "anthropic result"


async def test_provider_routing_openai() -> None:
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    with patch.object(
        client, "_complete_openai", new_callable=AsyncMock, return_value="openai result"
    ) as mock_method:
        result = await client.complete("hi")

    mock_method.assert_awaited_once()
    assert result == "openai result"


async def test_provider_routing_ollama() -> None:
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    with patch.object(
        client, "_complete_ollama", new_callable=AsyncMock, return_value="ollama result"
    ) as mock_method:
        result = await client.complete("hi")

    mock_method.assert_awaited_once()
    assert result == "ollama result"


async def test_unsupported_provider() -> None:
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    with (
        patch.object(client.settings, "provider", "unknown"),
        pytest.raises(ValueError, match="Unsupported provider"),
    ):
        await client.complete("hi")


async def test_ollama_completion() -> None:
    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "ollama generated text"}

    mock_http_client_class = MagicMock()
    mock_client_instance = mock_http_client_class.return_value
    mock_client_instance.post = AsyncMock(return_value=mock_response)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("furrow.llm.httpx.AsyncClient", mock_http_client_class):
        result = await client._complete_ollama("prompt text", "system text", "llama3")

    mock_client_instance.post.assert_awaited_once()
    call_args = mock_client_instance.post.call_args
    endpoint = call_args[0][0]
    assert endpoint == "http://localhost:11434/api/generate"
    payload = call_args[1]["json"]
    assert payload["model"] == "llama3"
    assert payload["prompt"] == "prompt text"
    assert payload["system"] == "system text"
    assert payload["stream"] is False
    assert result == "ollama generated text"


def test_anthropic_client_lazy_init(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(settings=Settings(anthropic_api_key=None))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        _ = client.anthropic
