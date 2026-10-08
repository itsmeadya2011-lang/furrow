from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from openai import AsyncOpenAI

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def ollama_client() -> LLMClient:
    return LLMClient(settings=Settings(provider=Provider.OLLAMA))


async def test_ollama_routing(ollama_client: LLMClient):
    ollama_client._complete_ollama = AsyncMock(return_value="ok")

    result = await ollama_client.complete("hello")

    assert result == "ok"
    ollama_client._complete_ollama.assert_awaited_once()


async def test_ollama_no_value_error(ollama_client: LLMClient):
    ollama_client._complete_ollama = AsyncMock(return_value="ok")

    result = await ollama_client.complete("hello")

    assert result == "ok"


def test_ollama_property_type(ollama_client: LLMClient):
    assert isinstance(ollama_client.ollama, AsyncOpenAI)


async def test_anthropic_retry_on_error():
    client = LLMClient(
        settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")
    )
    anthropic_mock = MagicMock()
    anthropic_mock.messages.create = AsyncMock(side_effect=ConnectionError("connection failed"))

    with (
        patch.object(LLMClient, "anthropic", new_callable=PropertyMock) as mock_prop,
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        mock_prop.return_value = anthropic_mock
        with pytest.raises(ConnectionError):
            await client.complete("hello")

    assert anthropic_mock.messages.create.call_count == 3
    assert mock_sleep.call_count == 2
