"""Tests for the LLMClient including provider routing and Ollama support."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def mock_settings():
    """Create a Settings instance with dummy API keys for testing."""
    return Settings(
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
        ollama_base_url="http://localhost:11434",
    )


@pytest.fixture
def mock_ollama_response():
    return {
        "model": "llama3",
        "message": {"role": "assistant", "content": "Hello from Ollama!"},
        "done": True,
    }


def _no_retry_decorator():
    """A retry decorator that doesn't actually retry (for faster tests)."""

    def decorator(func):
        return func

    return decorator


@pytest.mark.asyncio
async def test_complete_anthropic_routing(mock_settings):
    """Verify complete() routes to _complete_anthropic for Provider.ANTHROPIC."""
    mock_settings.provider = Provider.ANTHROPIC
    client = LLMClient(settings=mock_settings)

    with patch.object(
        client, "_complete_anthropic", new_callable=AsyncMock
    ) as mock_method:
        mock_method.return_value = "anthropic result"
        result = await client.complete("test prompt", system="test system")

        mock_method.assert_called_once()
        assert result == "anthropic result"


@pytest.mark.asyncio
async def test_complete_openai_routing(mock_settings):
    """Verify complete() routes to _complete_openai for Provider.OPENAI."""
    mock_settings.provider = Provider.OPENAI
    client = LLMClient(settings=mock_settings)

    with patch.object(
        client, "_complete_openai", new_callable=AsyncMock
    ) as mock_method:
        mock_method.return_value = "openai result"
        result = await client.complete("test prompt")

        mock_method.assert_called_once()
        assert result == "openai result"


@pytest.mark.asyncio
async def test_complete_ollama_routing(mock_settings):
    """Verify complete() routes to _complete_ollama for Provider.OLLAMA."""
    mock_settings.provider = Provider.OLLAMA
    client = LLMClient(settings=mock_settings)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": "ollama result"}
    }
    mock_response.raise_for_status = MagicMock()

    mock_http_client = MagicMock()
    mock_http_client.post = AsyncMock(return_value=mock_response)
    mock_http_client.aclose = AsyncMock()

    with patch.object(client, "_httpx", mock_http_client):
        result = await client.complete("test prompt", model="llama3")

    assert result == "ollama result"


@pytest.mark.asyncio
async def test_complete_unsupported_provider(mock_settings):
    """Verify complete() raises ValueError for unsupported providers."""
    mock_settings.provider = "invalid"
    client = LLMClient(settings=mock_settings)

    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("test prompt")


@pytest.mark.asyncio
async def test_complete_ollama_returns_content(mock_settings, mock_ollama_response):
    """Test the _complete_ollama method returns message content."""
    mock_settings.provider = Provider.OLLAMA
    client = LLMClient(settings=mock_settings)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_ollama_response
    mock_response.raise_for_status = MagicMock()

    mock_http_client = MagicMock()
    mock_http_client.post = AsyncMock(return_value=mock_response)
    mock_http_client.aclose = AsyncMock()

    # Patch retry to no-op for faster testing
    with patch.object(
        client, "_retry_decorator", _no_retry_decorator
    ), patch.object(client, "_httpx", mock_http_client):
        result = await client._complete_ollama("Hello", "", "llama3")

    assert result == "Hello from Ollama!"
    mock_http_client.post.assert_called_once()
    call_args = mock_http_client.post.call_args
    assert call_args[0][0] == "/api/chat"


@pytest.mark.asyncio
async def test_complete_ollama_empty_response(mock_settings):
    """Test _complete_ollama raises ValueError on empty content."""
    mock_settings.provider = Provider.OLLAMA
    client = LLMClient(settings=mock_settings)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"message": {"content": ""}}
    mock_response.raise_for_status = MagicMock()

    mock_http_client = MagicMock()
    mock_http_client.post = AsyncMock(return_value=mock_response)
    mock_http_client.aclose = AsyncMock()

    with patch.object(
        client, "_retry_decorator", _no_retry_decorator
    ), patch.object(client, "_httpx", mock_http_client), pytest.raises(
        ValueError, match="Ollama returned empty response"
    ):
        await client._complete_ollama("Hello", "", "llama3")


@pytest.mark.asyncio
async def test_read_file(mock_settings, tmp_path):
    """Test read_file returns file contents."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    client = LLMClient(settings=mock_settings)
    result = await client.read_file(str(test_file))
    assert result == "hello world"


@pytest.mark.asyncio
async def test_write_file(mock_settings, tmp_path):
    """Test write_file creates directories and writes content."""
    test_file = tmp_path / "subdir" / "output.txt"

    client = LLMClient(settings=mock_settings)
    await client.write_file(str(test_file), "file content")

    assert test_file.exists()
    assert test_file.read_text() == "file content"


def test_list_files(mock_settings, tmp_path):
    """Test list_files returns relative paths of files in a directory."""
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("c")

    client = LLMClient(settings=mock_settings)
    files = client.list_files(tmp_path)

    assert "a.py" in files
    assert "b.py" in files
    # Check for os-specific path separator
    c_path = os.path.join("sub", "c.py")
    assert c_path in files


def test_list_files_nonexistent(mock_settings):
    """Test list_files returns empty list for nonexistent directory."""
    client = LLMClient(settings=mock_settings)
    assert client.list_files("/nonexistent/path/xyz") == []


@pytest.mark.asyncio
async def test_retry_decorator_uses_tenacity(mock_settings):
    """Verify the retry decorator uses tenacity and retries on failure."""
    client = LLMClient(settings=mock_settings)
    decorator = client._retry_decorator()

    calls = []

    @decorator
    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("transient error")
        return "success"

    # Patch asyncio.sleep to avoid actual delays during retry
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await flaky()

    assert result == "success"
    assert len(calls) == 3
