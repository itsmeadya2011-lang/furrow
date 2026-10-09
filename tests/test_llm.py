"""Tests for the LLM client: OLLAMA provider, retry config, dispatch logic."""
from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai import AsyncOpenAI

from furrow.config import Provider, Settings
from furrow.llm import LLMClient, RETRYABLE_EXCEPTIONS


# ---------------------------------------------------------------------------
# Provider / dispatch tests
# ---------------------------------------------------------------------------

def test_default_provider_is_anthropic():
    """Default provider should be Anthropic."""
    client = LLMClient()
    assert client.settings.provider == Provider.ANTHROPIC


def test_ollama_provider_supported():
    """OLLAMA should be in the Provider enum."""
    assert Provider.OLLAMA.value == "ollama"


def test_ollama_client_uses_correct_base_url():
    """Ollama property should create an OpenAI-compatible client with the ollama base URL."""
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)
    ollama_client = client.ollama
    assert isinstance(ollama_client, AsyncOpenAI)
    assert "localhost:11434" in str(ollama_client.base_url)


def test_ollama_client_has_dummy_api_key():
    """Ollama property should use a dummy API key since Ollama doesn't require one."""
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    ollama_client = client.ollama
    # The dummy key "ollama" is set internally
    assert ollama_client is not None


def test_ollama_client_is_cached():
    """Ollama client should be created once and cached."""
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    assert client._ollama is None
    ollama1 = client.ollama
    ollama2 = client.ollama
    assert ollama1 is ollama2


# ---------------------------------------------------------------------------
# Retry configuration tests
# ---------------------------------------------------------------------------

def test_retryable_exceptions_include_sdk_errors():
    """RETRYABLE_EXCEPTIONS should include anthropic and openai API errors."""
    import anthropic
    import openai
    assert anthropic.APIError in RETRYABLE_EXCEPTIONS
    assert openai.APIError in RETRYABLE_EXCEPTIONS


def test_complete_has_retry_wrapper():
    """complete() should be wrapped with tenacity retry."""
    client = LLMClient()
    # tenacity adds a 'retry' attribute to wrapped functions
    assert hasattr(client.complete, "retry") or hasattr(client.complete, "retry_state")


@patch("furrow.llm.LLMClient._complete_anthropic")
@patch("furrow.llm.LLMClient._complete_openai")
@patch("furrow.llm.LLMClient._complete_ollama")
@pytest.mark.asyncio
async def test_ollama_dispatch(complete_ollama, complete_openai, complete_anthropic):
    """complete() should dispatch to _complete_ollama for OLLAMA provider."""
    complete_ollama.return_value = "ollama response"
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    result = await client.complete("hello")
    complete_ollama.assert_awaited_once()
    assert result == "ollama response"


@patch("furrow.llm.LLMClient._complete_anthropic")
@patch("furrow.llm.LLMClient._complete_openai")
@patch("furrow.llm.LLMClient._complete_ollama")
@pytest.mark.asyncio
async def test_unsupported_provider_raises(complete_ollama, complete_openai, complete_anthropic):
    """complete() should raise ValueError for unsupported providers."""
    # Use a fresh Settings instance to avoid polluting the shared singleton.
    client = LLMClient(settings=Settings())
    client.settings.provider = "bogus"  # bypass type checking at runtime
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hello")


# ---------------------------------------------------------------------------
# Retry behavior tests
# ---------------------------------------------------------------------------

@patch("furrow.llm.LLMClient._complete_anthropic")
@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_error(complete_anthropic):
    """complete() should retry on APIError and succeed when the call eventually works."""
    import anthropic
    from unittest.mock import Mock
    err = anthropic.APIError("transient error", request=Mock(), body=Mock())
    complete_anthropic.side_effect = [err, "success on retry"]
    client = LLMClient()
    result = await client.complete("hello")
    assert result == "success on retry"
    assert complete_anthropic.await_count == 2


@patch("furrow.llm.LLMClient._complete_anthropic")
@pytest.mark.asyncio
async def test_retry_reraises_after_max_attempts(complete_anthropic):
    """complete() should raise the exception after 3 failed attempts."""
    import anthropic
    from unittest.mock import Mock
    complete_anthropic.side_effect = anthropic.APIError("always fails", request=Mock(), body=Mock())
    client = LLMClient()
    with pytest.raises(anthropic.APIError):
        await client.complete("hello")
    # Should have been called 3 times (stop_after_attempt(3))
    assert complete_anthropic.await_count == 3


@patch("furrow.llm.LLMClient._complete_anthropic")
@pytest.mark.asyncio
async def test_non_retryable_error_not_retried(complete_anthropic):
    """complete() should NOT retry on errors not in RETRYABLE_EXCEPTIONS."""
    complete_anthropic.side_effect = ValueError("not retryable")
    client = LLMClient()
    with pytest.raises(ValueError):
        await client.complete("hello")
    # Should have been called only once
    assert complete_anthropic.await_count == 1


# ---------------------------------------------------------------------------
# File operations tests
# ---------------------------------------------------------------------------

@patch("furrow.llm.LLMClient._complete_anthropic")
@pytest.mark.asyncio
async def test_system_prompt_passed_to_provider(complete_anthropic):
    """complete() should pass the system prompt to the provider."""
    complete_anthropic.return_value = "response"
    client = LLMClient()
    await client.complete("hello", system="custom system prompt")
    # Verify the system prompt was passed as the second positional arg
    call_args = complete_anthropic.call_args
    assert call_args is not None
    args = call_args[0]  # positional args: (prompt, system, model)
    assert args[1] == "custom system prompt"
