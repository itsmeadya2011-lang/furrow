import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from anthropic import RateLimitError

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_get_timeout_default():
    settings = Settings()
    client = LLMClient(settings=settings)
    assert client._get_timeout() == 120.0


def test_get_timeout_custom():
    settings = Settings(request_timeout=60)
    client = LLMClient(settings=settings)
    assert client._get_timeout() == 60.0


def test_complete_raises_value_error_unsupported_provider():
    settings = Settings(provider=Provider("unsupported"))
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="Unsupported provider"):
        asyncio.run(client.complete("test"))


def test_complete_anthropic_retry_on_rate_limit():
    settings = Settings(
        provider=Provider.ANTHROPIC,
        retry_attempts=3,
        retry_backoff=0.1,
        anthropic_api_key="test-key",
    )
    client = LLMClient(settings=settings)

    mock_response = AsyncMock()
    mock_response.content = [AsyncMock(text="mocked response")]

    with patch.object(client, "anthropic") as mock_anthropic:
        mock_anthropic.messages.create = AsyncMock(
            side_effect=RateLimitError("rate limited")
        )
        with pytest.raises(RuntimeError, match="Anthropic completion failed"):
            asyncio.run(
                client._complete_anthropic("test", "system", "model", 0.7, 100)
            )
