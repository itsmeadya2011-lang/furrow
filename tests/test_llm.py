from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def make_ollama_settings() -> Settings:
    return Settings(provider=Provider.OLLAMA, model="llama3", openai_api_key="x")


def make_ollama_client() -> LLMClient:
    return LLMClient(settings=make_ollama_settings())


def make_fake_ollama():
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "ollama response"

    create_mock = AsyncMock(return_value=fake_response)
    chat_mock = MagicMock()
    chat_mock.completions = MagicMock()
    chat_mock.completions.create = create_mock

    ollama_client_mock = MagicMock()
    ollama_client_mock.chat = chat_mock
    return ollama_client_mock, create_mock


@pytest.mark.asyncio
async def test_complete_routes_through_ollama():
    client = make_ollama_client()
    ollama_client_mock, create_mock = make_fake_ollama()
    client._ollama = ollama_client_mock

    result = await client.complete(prompt="hi", system="", model="llama3")

    create_mock.assert_awaited_once()
    kwargs = create_mock.call_args.kwargs
    assert kwargs["model"] == "llama3"
    assert kwargs["messages"] == [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "hi"},
    ]
    assert result == "ollama response"


@pytest.mark.asyncio
async def test_complete_ollama_uses_custom_system():
    client = make_ollama_client()
    ollama_client_mock, create_mock = make_fake_ollama()
    client._ollama = ollama_client_mock

    await client.complete(prompt="hi", system="be brief", model="llama3")

    kwargs = create_mock.call_args.kwargs
    assert kwargs["messages"][0]["content"] == "be brief"


@pytest.mark.asyncio
async def test_ollama_property_lazily_caches():
    client = LLMClient(settings=make_ollama_settings())
    assert client._ollama is None
    first = client.ollama
    second = client.ollama
    assert first is second
    assert client._ollama is first


@pytest.mark.asyncio
async def test_ollama_property_uses_base_url():
    client = LLMClient(
        settings=Settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:9999",
            openai_api_key="x",
        )
    )
    instance = client.ollama
    assert instance.base_url == "http://localhost:9999"


@pytest.mark.asyncio
async def test_complete_unsupported_provider():
    client = LLMClient(settings=SimpleNamespace(provider="bogus", model="x"))
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete(prompt="hi")
