from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


async def test_ollama_base_url():
    s = Settings(_env_file=None, provider=Provider.OLLAMA)
    client = LLMClient(settings=s)
    assert str(client.ollama.base_url).rstrip("/") == "http://localhost:11434/v1"


async def test_empty_content_raises():
    s = Settings(_env_file=None, provider=Provider.ANTHROPIC)
    client = LLMClient(settings=s)
    mock_response = AsyncMock()
    mock_response.content = []
    client._anthropic = AsyncMock()
    client._anthropic.messages.create = AsyncMock(
        return_value=mock_response
    )
    with pytest.raises(RuntimeError, match="LLM returned empty content"):
        await client._complete_anthropic("prompt", "", "model")


async def test_complete_routes_ollama():
    s = Settings(_env_file=None, provider=Provider.OLLAMA)
    client = LLMClient(settings=s)
    mock_response = AsyncMock()
    mock_response.choices[0].message.content = "hello"
    client._ollama = AsyncMock()
    client._ollama.chat.completions.create = AsyncMock(
        return_value=mock_response
    )
    result = await client.complete("prompt")
    assert result == "hello"


async def test_read_write_file(tmp_path):
    s = Settings(_env_file=None)
    client = LLMClient(settings=s)
    path = tmp_path / "a.txt"
    await client.write_file(path, "hi")
    assert await client.read_file(path) == "hi"
