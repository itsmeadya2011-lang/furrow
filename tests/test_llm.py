import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from furrow.llm import (
    LLMError, LLMConnectionError, LLMRateLimitError, LLMTimeoutError,
    _is_retryable_exception, _classify_exception
)

class TestLLMExceptions:
    def test_exception_hierarchy(self):
        assert issubclass(LLMConnectionError, LLMError)
        assert issubclass(LLMRateLimitError, LLMError)
        assert issubclass(LLMTimeoutError, LLMError)

class TestIsRetryableException:
    def test_connection_error_is_retryable(self):
        from anthropic import APIConnectionError
        assert _is_retryable_exception(APIConnectionError("fail")) is True

    def test_timeout_error_is_retryable(self):
        assert _is_retryable_exception(TimeoutError("timeout")) is True

    def test_rate_limit_is_retryable(self):
        from anthropic import RateLimitError
        assert _is_retryable_exception(RateLimitError("rate")) is True

    def test_5xx_error_is_retryable(self):
        from anthropic import APIStatusError
        err = MagicMock(spec=APIStatusError)
        err.status_code = 503
        assert _is_retryable_exception(err) is True

    def test_4xx_error_not_retryable(self):
        from anthropic import APIStatusError
        err = MagicMock(spec=APIStatusError)
        err.status_code = 400
        assert _is_retryable_exception(err) is False

class TestClassifyException:
    def test_timeout_classified_correctly(self):
        err = TimeoutError("timeout")
        result = _classify_exception(err)
        assert isinstance(result, LLMTimeoutError)

    def test_connection_error_classified_correctly(self):
        err = ConnectionError("conn")
        result = _classify_exception(err)
        assert isinstance(result, LLMConnectionError)

class TestLLMClientRetry:
    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        from furrow.llm import LLMClient
        from furrow.config import Settings, Provider

        settings = MagicMock()
        settings.provider = Provider.ANTHROPIC
        settings.model = "test-model"

        client = LLMClient(settings=settings, max_retries=2, retry_min_wait=0.01, retry_max_wait=0.01)

        # Mock anthropic to fail once then succeed
        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(side_effect=[
            Exception("transient"),  # First call fails
            MagicMock(content=[MagicMock(text="success")])  # Second call succeeds
        ])
        client._anthropic = mock_anthropic

        result = await client.complete("test prompt")
        assert result == "success"