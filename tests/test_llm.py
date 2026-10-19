import pytest
from unittest.mock import AsyncMock, MagicMock

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.mark.asyncio
async def test_ollama_provider_dispatch():
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="hi"))]))
    client._ollama = fake
    # monkeypatch property by setting _ollama already set; property returns it
    res = await client.complete("prompt")
    assert res == "hi"
    fake.chat.completions.create.assert_awaited()


@pytest.mark.asyncio
async def test_openai_provider_dispatch():
    settings = Settings(provider=Provider.OPENAI, openai_api_key="key")
    client = LLMClient(settings=settings)
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="hello"))]))
    client._openai = fake
    res = await client.complete("prompt")
    assert res == "hello"
    fake.chat.completions.create.assert_awaited()