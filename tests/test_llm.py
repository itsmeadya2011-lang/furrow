import asyncio
from pathlib import Path
from anthropic.types import TextBlock
from unittest.mock import AsyncMock, Mock, patch

import pytest

from furrow.llm import (
    LLMAuthError,
    LLMClient,
    LLMError,
    LLMRateLimitError,
)
from furrow.config import Provider, Settings


def test_llm_client_init():
    settings = Settings(_env_file=None)
    client = LLMClient(settings=settings)
    assert client.settings == settings
    assert client._anthropic is None
    assert client._openai is None
    assert client._httpx is None


@pytest.mark.asyncio
async def test_complete_anthropic_with_mock():
    settings = Settings(_env_file=None)
    settings.provider = Provider.ANTHROPIC
    client = LLMClient(settings=settings)
    client._anthropic = Mock()

    mock_response = Mock()
    mock_content = Mock(spec=TextBlock)
    mock_content.text = "Hello from Anthropic"
    mock_response.content = [mock_content]

    client._anthropic.messages.create = AsyncMock(return_value=mock_response)
    result = await client.complete("Test prompt", "Test system")
    assert result == "Hello from Anthropic"


@pytest.mark.asyncio
async def test_complete_openai_with_mock():
    settings = Settings(_env_file=None)
    settings.provider = Provider.OPENAI
    client = LLMClient(settings=settings)
    client._openai = Mock()

    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "Hello from OpenAI"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    client._openai.chat.completions.create = AsyncMock(return_value=mock_response)
    result = await client.complete("Test prompt")
    assert result == "Hello from OpenAI"


@pytest.mark.asyncio
async def test_complete_unsupported_provider():
    settings = Settings(_env_file=None)

    class FakeProvider:
        value = "unsupported"

    settings.provider = FakeProvider()  # type: ignore
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("Test prompt")


@pytest.mark.asyncio
async def test_read_write_file(tmp_path):
    settings = Settings(_env_file=None)
    client = LLMClient(settings=settings)
    test_file = tmp_path / "test.txt"
    content = "Hello, World!"
    await client.write_file(test_file, content)
    read_content = await client.read_file(test_file)
    assert read_content == content


def test_list_files(tmp_path):
    settings = Settings(_env_file=None)
    client = LLMClient(settings=settings)
    (tmp_path / "file1.txt").write_text("1")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file2.py").write_text("2")
    files = client.list_files(tmp_path)
    assert "file1.txt" in files
    assert "subdir/file2.py" in files


def test_llm_error_hierarchy():
    assert issubclass(LLMAuthError, LLMError)
    assert issubclass(LLMRateLimitError, LLMError)
    assert issubclass(LLMError, Exception)


@pytest.mark.asyncio
async def test_complete_with_tools_anthropic():
    settings = Settings(_env_file=None)
    settings.provider = Provider.ANTHROPIC
    client = LLMClient(settings=settings)
    client._anthropic = Mock()

    mock_response = Mock()
    mock_response.content = []

    tools = [
        {
            "name": "read_file",
            "description": "Read file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]

    client._anthropic.messages.create = AsyncMock(return_value=mock_response)
    result = await client.complete_with_tools("prompt", tools, "system")
    assert result == mock_response


@pytest.mark.asyncio
async def test_complete_with_tools_openai():
    settings = Settings(_env_file=None)
    settings.provider = Provider.OPENAI
    client = LLMClient(settings=settings)
    client._openai = Mock()

    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    tools = [
        {
            "name": "read_file",
            "description": "Read file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]

    client._openai.chat.completions.create = AsyncMock(return_value=mock_response)
    result = await client.complete_with_tools("prompt", tools, "system")
    assert result == mock_response


@pytest.mark.asyncio
async def test_complete_with_tools_ollama_raises():
    settings = Settings(_env_file=None)
    settings.provider = Provider.OLLAMA
    client = LLMClient(settings=settings)

    with pytest.raises(NotImplementedError, match="Ollama does not support tool use"):
        await client.complete_with_tools("prompt", [], "system")
