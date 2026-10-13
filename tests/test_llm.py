"""Tests for the LLMClient, including multi-provider support."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_default_provider():
    """Default provider should be Anthropic."""
    client = LLMClient()
    assert client.settings.provider == Provider.ANTHROPIC


def test_provider_override():
    """Provider should be configurable via Settings."""
    s = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=s)
    assert client.settings.provider == Provider.OPENAI


def test_ollama_provider_config():
    """Ollama provider should use the ollama_base_url setting."""
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)
    assert client.settings.provider == Provider.OLLAMA
    assert client.settings.ollama_base_url == "http://localhost:11434"


@pytest.mark.asyncio
async def test_complete_uses_default_model():
    """complete() should use settings.model when no model is provided."""
    s = Settings(provider=Provider.ANTHROPIC, model="claude-test", anthropic_api_key="test-key")
    client = LLMClient(settings=s)

    with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock:
        mock.return_value = "response"
        await client.complete("prompt")
        # Check that model="claude-test" was passed
        assert "claude-test" in str(mock.call_args)


@pytest.mark.asyncio
async def test_complete_routes_to_openai():
    """complete() should route to _complete_openai for OPENAI provider."""
    s = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=s)

    with patch.object(client, "_complete_openai", new_callable=AsyncMock) as mock:
        mock.return_value = "response"
        result = await client.complete("prompt", system="sys")
        assert result == "response"
        mock.assert_called_once()


@pytest.mark.asyncio
async def test_complete_routes_to_anthropic():
    """complete() should route to _complete_anthropic for ANTHROPIC provider."""
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient(settings=s)

    with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock:
        mock.return_value = "response"
        result = await client.complete("prompt", system="sys")
        assert result == "response"
        mock.assert_called_once()


@pytest.mark.asyncio
async def test_complete_routes_to_ollama():
    """complete() should route to _complete_ollama for OLLAMA provider."""
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)

    with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock:
        mock.return_value = "response"
        result = await client.complete("prompt", system="sys")
        assert result == "response"
        mock.assert_called_once()


@pytest.mark.asyncio
async def test_ollama_complete_format():
    """_complete_ollama should parse the Ollama chat response format."""
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)

    # Pre-populate the httpx client with a mock
    mock_httpx = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "model": "llama3",
        "message": {"role": "assistant", "content": "Hello from Ollama"},
    }
    mock_httpx.post = AsyncMock(return_value=mock_response)
    client._httpx = mock_httpx

    result = await client._complete_ollama("prompt", "system", "llama3")
    assert result == "Hello from Ollama"
    mock_httpx.post.assert_called_once()


@pytest.mark.asyncio
async def test_ollama_error_handling():
    """_complete_ollama should raise for HTTP errors."""
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)

    mock_httpx = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
    )
    mock_response.json.return_value = {}
    mock_httpx.post = AsyncMock(return_value=mock_response)
    client._httpx = mock_httpx

    with pytest.raises(httpx.HTTPStatusError):
        await client._complete_ollama("prompt", "", "llama3")


@pytest.mark.asyncio
async def test_anthropic_missing_api_key():
    """Should raise ValueError when ANTHROPIC_API_KEY is not set."""
    s = Settings(
        provider=Provider.ANTHROPIC,
        anthropic_api_key=None,
    )
    client = LLMClient(settings=s)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            _ = client.anthropic


@pytest.mark.asyncio
async def test_openai_missing_api_key():
    """Should raise ValueError when OPENAI_API_KEY is not set."""
    s = Settings(
        provider=Provider.OPENAI,
        openai_api_key=None,
    )
    client = LLMClient(settings=s)
    with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            _ = client.openai


@pytest.mark.asyncio
async def test_list_files(tmp_path):
    """list_files should return relative file paths."""
    s = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=s)

    (tmp_path / "file1.txt").write_text("hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "file2.txt").write_text("world")

    files = client.list_files(tmp_path)
    assert "file1.txt" in files
    assert "sub/file2.txt" in files or "sub\\file2.txt" in files


@pytest.mark.asyncio
async def test_list_files_nonexistent():
    """list_files should return empty list for nonexistent directory."""
    s = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=s)
    assert client.list_files("/nonexistent/path") == []


@pytest.mark.asyncio
async def test_read_write_file(tmp_path):
    """read_file and write_file should work correctly."""
    s = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=s)

    test_file = tmp_path / "test.txt"
    await client.write_file(test_file, "test content")
    content = await client.read_file(test_file)
    assert content == "test content"


@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(tmp_path):
    """write_file should create parent directories as needed."""
    s = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=s)

    nested_file = tmp_path / "nested" / "deep" / "file.txt"
    await client.write_file(nested_file, "nested content")
    content = await client.read_file(nested_file)
    assert content == "nested content"


@pytest.mark.asyncio
async def test_aclose_closes_httpx():
    """aclose should close the httpx client."""
    s = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=s)

    # Initialize the httpx client
    _ = client.httpx_client
    assert client._httpx is not None

    # Set a mock with an async aclose
    mock_httpx = MagicMock()
    mock_httpx.aclose = AsyncMock()
    client._httpx = mock_httpx
    await client.aclose()
    mock_httpx.aclose.assert_called_once()
    assert client._httpx is None


@pytest.mark.asyncio
async def test_init_with_none_settings():
    """LLMClient should handle settings=None by creating default Settings."""
    client = LLMClient(settings=None)
    assert client.settings is not None
    assert client.settings.provider == Provider.ANTHROPIC


@pytest.mark.asyncio
async def test_settings_propagation():
    """Orchestrator should propagate settings to the LLMClient."""
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://custom:1234")
    client = LLMClient(settings=s)
    assert client.settings.ollama_base_url == "http://custom:1234"
