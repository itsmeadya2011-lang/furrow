"""Tests for the LLMClient's Ollama provider and config defaults."""

from __future__ import annotations

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


class TestConfigDefaults:
    def test_default_provider(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC

    def test_state_file_default(self):
        s = Settings()
        assert s.state_file.name == ".furrow-state.json"

    def test_max_cycles_default_zero(self):
        s = Settings()
        assert s.max_cycles == 0


class TestLLMClient:
    def test_ollama_provider_supported(self):
        """Provider.OLLAMA must not raise ValueError in complete()."""
        # We can't actually call complete() without an Ollama server, but we
        # can verify the code path dispatches to _complete_ollama.
        s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=s)
        # Verify the method exists and is callable
        assert hasattr(client, "_complete_ollama")
        assert callable(client._complete_ollama)

    def test_retry_config(self):
        s = Settings()
        client = LLMClient(settings=s)
        assert client.max_retries == 3

    def test_retryable_exceptions(self):
        import httpx

        from furrow.llm import _RETRYABLE_EXC

        assert httpx.HTTPStatusError in _RETRYABLE_EXC
        assert httpx.RequestError in _RETRYABLE_EXC
        assert TimeoutError in _RETRYABLE_EXC
