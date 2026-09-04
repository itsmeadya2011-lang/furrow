import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def client():
    s = Settings(provider=Provider.ANTHROPIC, model="test-model")
    return LLMClient(settings=s)


async def test_complete_anthropic(client):
    with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock_method:
        mock_method.return_value = "anthropic response"
        result = await client.complete("test prompt")
        assert result == "anthropic response"
        mock_method.assert_called_once_with("test prompt", "", "test-model")


async def test_complete_openai(client):
    client.settings.provider = Provider.OPENAI
    with patch.object(client, "_complete_openai", new_callable=AsyncMock) as mock_method:
        mock_method.return_value = "openai response"
        result = await client.complete("test prompt")
        assert result == "openai response"
        mock_method.assert_called_once_with("test prompt", "", "test-model")


async def test_complete_ollama(client):
    client.settings.provider = Provider.OLLAMA
    with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock_method:
        mock_method.return_value = "ollama response"
        result = await client.complete("test prompt")
        assert result == "ollama response"
        mock_method.assert_called_once_with("test prompt", "", "test-model")


async def test_complete_unsupported_provider(client):
    client.settings.provider = "unsupported"
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("test prompt")


async def test_complete_uses_default_model(client):
    with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock_method:
        mock_method.return_value = "response"
        await client.complete("test prompt", model=None)
        mock_method.assert_called_once_with("test prompt", "", "test-model")


async def test_ollama_streaming_response():
    s = Settings(provider=Provider.OLLAMA, model="llama-test", ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()

    async def mock_aiter():
        yield '{"response": "ollama says hi"}'
        yield ""

    mock_response.aiter_lines.return_value = mock_aiter()

    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream.__aexit__ = AsyncMock(return_value=False)

    mock_http_client = AsyncMock()
    mock_http_client.stream.return_value = mock_stream
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    with patch("furrow.llm.httpx.AsyncClient", return_value=mock_http_client):
        result = await client._complete_ollama("prompt", "", "llama-test")

    assert result == "ollama says hi"


async def test_ollama_raises_on_empty_response():
    s = Settings(provider=Provider.OLLAMA, model="llama-test", ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()

    async def mock_aiter():
        yield ""
        yield "   "

    mock_response.aiter_lines.return_value = mock_aiter()

    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream.__aexit__ = AsyncMock(return_value=False)

    mock_http_client = AsyncMock()
    mock_http_client.stream.return_value = mock_stream
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    with patch("furrow.llm.httpx.AsyncClient", return_value=mock_http_client):
        with pytest.raises(ValueError, match="No response received from Ollama"):
            await client._complete_ollama("prompt", "", "llama-test")


async def test_read_file():
    client = LLMClient()
    mock_file = AsyncMock()
    mock_file.read.return_value = "file content"

    with patch("furrow.llm.aiofiles.open", return_value=mock_file):
        result = await client.read_file("/fake/path.txt")

    assert result == "file content"


async def test_write_file():
    client = LLMClient()
    mock_file = AsyncMock()

    with patch("furrow.llm.aiofiles.open", return_value=mock_file), patch.object(Path, "mkdir") as mock_mkdir:
        await client.write_file("/fake/dir/file.txt", "content")

    mock_file.write.assert_called_once_with("content")
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


async def test_write_file_creates_parent_dirs():
    client = LLMClient()
    mock_file = AsyncMock()

    with patch("furrow.llm.aiofiles.open", return_value=mock_file) as mock_open, \
         patch.object(Path, "mkdir") as mock_mkdir:
        await client.write_file("/fake/nested/dir/file.txt", "content")

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_open.assert_called_once()


def test_list_files_existing_dir():
    client = LLMClient()

    mock_file1 = MagicMock()
    mock_file1.is_file.return_value = True
    mock_file1.relative_to.return_value = Path("file1.txt")

    mock_file2 = MagicMock()
    mock_file2.is_file.return_value = True
    mock_file2.relative_to.return_value = Path("subdir/file2.txt")

    mock_dir = MagicMock()
    mock_dir.exists.return_value = True
    mock_dir.rglob.return_value = [mock_file1, mock_file2]

    with patch("furrow.llm.Path", return_value=mock_dir):
        result = client.list_files("/fake/dir")

    assert result == ["file1.txt", "subdir/file2.txt"]


def test_list_files_nonexistent_dir():
    client = LLMClient()

    mock_dir = MagicMock()
    mock_dir.exists.return_value = False

    with patch("furrow.llm.Path", return_value=mock_dir):
        result = client.list_files("/fake/dir")

    assert result == []


async def test_anthropic_lazy_init():
    s = Settings(provider=Provider.ANTHROPIC, model="test", anthropic_api_key="test-key")
    client = LLMClient(settings=s)
    assert client._anthropic is None

    mock_anthropic_instance = MagicMock()
    with patch("furrow.llm.AsyncAnthropic", return_value=mock_anthropic_instance) as mock_async_anthropic:
        instance = client.anthropic

    assert client._anthropic is mock_anthropic_instance
    assert instance is mock_anthropic_instance
    mock_async_anthropic.assert_called_once_with(api_key="test-key")


async def test_anthropic_missing_api_key():
    s = Settings(provider=Provider.ANTHROPIC, model="test", anthropic_api_key=None)
    client = LLMClient(settings=s)

    with patch("furrow.llm.os.getenv", return_value=None):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            _ = client.anthropic
