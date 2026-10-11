from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.mark.asyncio
async def test_complete_dispatches_anthropic():
    client = LLMClient(Settings(provider=Provider.ANTHROPIC))
    with patch.object(client, "_complete_anthropic", new=AsyncMock(return_value="anthropic-response")) as mock_method:
        result = await client.complete("hello", model="claude-test")
        assert result == "anthropic-response"
        mock_method.assert_awaited_once_with("hello", "", "claude-test")


@pytest.mark.asyncio
async def test_complete_dispatches_openai():
    client = LLMClient(Settings(provider=Provider.OPENAI))
    with patch.object(client, "_complete_openai", new=AsyncMock(return_value="openai-response")) as mock_method:
        result = await client.complete("hello", model="gpt-test")
        assert result == "openai-response"
        mock_method.assert_awaited_once_with("hello", "", "gpt-test")


@pytest.mark.asyncio
async def test_complete_dispatches_ollama():
    client = LLMClient(Settings(provider=Provider.OLLAMA))
    with patch.object(client, "_complete_ollama", new=AsyncMock(return_value="ollama-response")) as mock_method:
        result = await client.complete("hello", model="llama-test")
        assert result == "ollama-response"
        mock_method.assert_awaited_once_with("hello", "", "llama-test")


@pytest.mark.asyncio
async def test_complete_defaults_to_settings_model():
    settings = Settings(provider=Provider.ANTHROPIC, model="default-model")
    client = LLMClient(settings)
    with patch.object(client, "_complete_anthropic", new=AsyncMock(return_value="ok")) as mock_method:
        await client.complete("hello")
        mock_method.assert_awaited_once_with("hello", "", "default-model")


@pytest.mark.asyncio
async def test_complete_unsupported_provider():
    # Force an unsupported provider by creating a client with a fake provider
    class FakeProvider:
        value = "bogus"

    client = LLMClient.__new__(LLMClient)
    client.settings = MagicMock()
    client.settings.provider = FakeProvider()
    client._anthropic = None
    client._openai = None
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hello")


@pytest.mark.asyncio
async def test_ollama_http_request():
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings)
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "ollama output"}
    mock_response.raise_for_status = MagicMock()

    with patch("furrow.llm.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await client._complete_ollama("prompt text", "system text", "llama3")
        assert result == "ollama output"

        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/generate"
        body = call_args[1]["json"]
        assert body["model"] == "llama3"
        assert body["prompt"] == "prompt text"
        assert body["system"] == "system text"
        assert body["stream"] is False


@pytest.mark.asyncio
async def test_ollama_http_error_raises_connection_error():
    import httpx

    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings)
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=MagicMock()
    )

    with patch("furrow.llm.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(ConnectionError, match="Ollama request failed"):
            await client._complete_ollama("p", "s", "m")


@pytest.mark.asyncio
async def test_retry_decorator_applied_on_failure():
    """Test that retry happens on failure using tenacity's retry info."""
    import tenacity

    # Check that the decorated method has retry state
    client = LLMClient(Settings(provider=Provider.ANTHROPIC))

    # Verify the method is wrapped by tenacity
    assert hasattr(client._complete_anthropic, "retry") or hasattr(client._complete_anthropic, "statistic")

    # Actually test retry behavior: make the underlying call fail N-1 times then succeed
    call_count = 0

    async def flaky_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise anthropic_mock_error()
        return success_response()

    mock_anthropic = MagicMock()
    mock_anthropic.messages.create = flaky_create
    client._anthropic = mock_anthropic

    result = await client._complete_anthropic("hello", "sys", "model-x")
    assert result == "success"
    assert call_count == 3


def anthropic_mock_error():
    """Create a real APIConnectionError that tenacity will retry on."""
    import anthropic
    import httpx2
    return anthropic.APIConnectionError(message="mock connection error", request=httpx2.Request("POST", "http://mock"))


def success_response():
    resp = MagicMock()
    content = MagicMock()
    content.text = "success"
    resp.content = [content]
    return resp