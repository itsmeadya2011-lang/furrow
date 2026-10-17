import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def settings():
    return Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")


@pytest.mark.asyncio
async def test_llm_client_anthropic(settings):
    client = LLMClient(settings=settings)
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "Hello from Anthropic"

    with patch.object(client.anthropic.__class__, "messages", create=True) as mock_messages:
        mock_messages.create = AsyncMock(return_value=mock_response)
        result = await client.complete("Say hello", model="claude-3")
        assert result == "Hello from Anthropic"


@pytest.mark.asyncio
async def test_llm_client_openai():
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello from OpenAI"

    with patch.object(client.openai.__class__, "chat", create=True) as mock_chat:
        mock_chat.completions.create = AsyncMock(return_value=mock_response)
        result = await client.complete("Say hello", model="gpt-4")
        assert result == "Hello from OpenAI"


@pytest.mark.asyncio
async def test_llm_client_ollama():
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"response": "Hello from Ollama"})
    mock_response.raise_for_status = MagicMock()

    with patch("furrow.llm.httpx.AsyncClient") as mock_client_class:
        mock_client_instance = MagicMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await client.complete("Say hello", model="llama3.1")
        assert result == "Hello from Ollama"


@pytest.mark.asyncio
async def test_llm_client_unsupported_provider():
    settings = Settings(provider=Provider.ANTHROPIC)
    # Temporarily set an invalid provider
    settings.provider = "invalid"
    client = LLMClient(settings=settings)

    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("test")


def test_llm_client_list_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("world")

    client = LLMClient()
    files = client.list_files(tmp_path)
    assert "a.txt" in files
    assert "sub/b.txt" in files


def test_llm_client_list_files_nonexistent():
    client = LLMClient()
    files = client.list_files("/nonexistent/path/12345")
    assert files == []
