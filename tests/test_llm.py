from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def make_ollama_client() -> LLMClient:
    return LLMClient(
        Settings(
            provider=Provider.OLLAMA,
            model="test-model",
            ollama_base_url="http://ollama.test",
        )
    )


@pytest.mark.asyncio
async def test_complete_routes_to_ollama_when_provider_is_ollama():
    client = make_ollama_client()
    complete_ollama = AsyncMock(return_value="ollama response")
    client._complete_ollama = complete_ollama

    result = await client.complete("prompt", system="system prompt", model="model")

    assert result == "ollama response"
    complete_ollama.assert_awaited_once_with("prompt", "system prompt", "model")


@pytest.mark.asyncio
async def test_ollama_response_parsing_extracts_response_field():
    client = make_ollama_client()
    response = httpx.Response(
        200,
        json={"response": "generated text"},
        request=httpx.Request("POST", "http://ollama.test/api/generate"),
    )
    post = AsyncMock(return_value=response)
    client._ollama = Mock(post=post)

    result = await client._complete_ollama("prompt", "system", "model")

    assert result == "generated text"
    post.assert_awaited_once_with(
        "/api/generate",
        json={
            "model": "model",
            "prompt": "prompt",
            "system": "system",
            "stream": False,
        },
    )


@pytest.mark.asyncio
async def test_ollama_connection_error_raises_value_error():
    client = make_ollama_client()
    client._ollama = Mock(
        post=AsyncMock(side_effect=httpx.ConnectError("connection failed"))
    )

    with pytest.raises(ValueError, match="Unable to connect to Ollama"):
        await client._complete_ollama("prompt", "system", "model")


@pytest.mark.asyncio
async def test_ollama_http_error_status_raises_value_error():
    client = make_ollama_client()
    response = httpx.Response(
        503,
        json={"error": "service unavailable"},
        request=httpx.Request("POST", "http://ollama.test/api/generate"),
    )
    client._ollama = Mock(post=AsyncMock(return_value=response))

    with pytest.raises(ValueError, match="status 503.*service unavailable"):
        await client._complete_ollama("prompt", "system", "model")


@pytest.mark.asyncio
async def test_ollama_invalid_json_raises_value_error():
    client = make_ollama_client()
    response = httpx.Response(
        200,
        text="not json",
        request=httpx.Request("POST", "http://ollama.test/api/generate"),
    )
    client._ollama = Mock(post=AsyncMock(return_value=response))

    with pytest.raises(ValueError, match="invalid JSON response"):
        await client._complete_ollama("prompt", "system", "model")


@pytest.mark.asyncio
async def test_ollama_missing_response_field_raises_value_error():
    client = make_ollama_client()
    response = httpx.Response(
        200,
        json={"model": "model"},
        request=httpx.Request("POST", "http://ollama.test/api/generate"),
    )
    client._ollama = Mock(post=AsyncMock(return_value=response))

    with pytest.raises(ValueError, match="missing the 'response' field"):
        await client._complete_ollama("prompt", "system", "model")
