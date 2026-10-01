"""Tests for LLM client provider routing, retry behavior, and file helpers.

All external API/LLM calls are mocked — no network access required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Provider, Settings
from furrow.llm import LLMClient

# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

class TestProviderRouting:
    """Verify LLMClient.complete dispatches to the right provider."""

    async def test_openai_provider_routes_correctly(self) -> None:
        s = Settings(provider=Provider.OPENAI, openai_api_key="sk-test")
        client = LLMClient(settings=s)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "openai response"

        with patch.object(type(client), "openai", new_callable=MagicMock) as mock_prop:
            mock_openai_client = AsyncMock()
            mock_openai_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            mock_prop.__get__ = MagicMock(return_value=mock_openai_client)

            result = await client.complete("hello", system="test", model="gpt-4")
            assert result == "openai response"

    async def test_unsupported_provider_raises(self) -> None:
        s = Settings(provider=Provider.ANTHROPIC)
        client = LLMClient(settings=s)
        with patch.object(client.settings, "provider", "invalid"):
            try:
                await client.complete("hi")
                raise AssertionError("Should have raised ValueError")
            except ValueError as e:
                assert "Unsupported provider" in str(e)

    async def test_ollama_model_name_strips_prefix(self) -> None:
        """Ollama model names may include an 'ollama/' prefix that should be stripped."""
        s = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=s)

        with patch("httpx.AsyncClient") as mock_http:
            mock_instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "model": "llama3",
                "response": "ollama response",
                "done": True,
            }
            mock_response.raise_for_status = MagicMock()
            mock_instance.post = AsyncMock(return_value=mock_response)

            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_http.return_value = mock_context

            result = await client.complete("hello", model="ollama/llama3")
            assert result == "ollama response"


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    """Verify tenacity retry logic kicks in on transient failures."""

    async def test_anthropic_retries_on_api_error(self) -> None:
        import anthropic

        s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="key")
        client = LLMClient(settings=s)

        mock_client = AsyncMock()

        success_response = MagicMock()
        success_response.content = [MagicMock(text="success after retry")]

        # Build a dummy request object required by anthropic.APIError
        dummy_request = MagicMock()

        mock_client.messages.create = AsyncMock(
            side_effect=[
                anthropic.APIError("transient", request=dummy_request, body=None),
                anthropic.APIError("transient", request=dummy_request, body=None),
                success_response,
            ]
        )
        with patch.object(type(client), "anthropic", new_callable=MagicMock) as mock_prop:
            mock_prop.__get__ = MagicMock(return_value=mock_client)
            result = await client.complete("hello", model="claude-3-5-sonnet-20241022")
            assert result == "success after retry"
            assert mock_client.messages.create.call_count == 3

    async def test_ollama_retries_on_http_error(self) -> None:
        import httpx

        s = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=s)

        success_payload = {"model": "llama3", "response": "ok", "done": True}

        with patch("httpx.AsyncClient") as mock_http:
            mock_instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = success_payload
            mock_response.raise_for_status = MagicMock()

            # First call raises a connection error, second succeeds
            mock_instance.post = AsyncMock(
                side_effect=[httpx.ConnectError("conn refused"), mock_response]
            )

            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_http.return_value = mock_context

            result = await client.complete("hello", model="llama3")
            assert result == "ok"
            assert mock_instance.post.call_count == 2


# ---------------------------------------------------------------------------
# LLMClient file helpers
# ---------------------------------------------------------------------------

class TestLLMFileHelpers:
    async def test_list_files_empty_dir(self, tmp_path) -> None:
        s = Settings()
        client = LLMClient(settings=s)
        result = client.list_files(tmp_path)
        assert result == []

    async def test_list_files_nonexistent_dir(self) -> None:
        s = Settings()
        client = LLMClient(settings=s)
        result = client.list_files("/nonexistent/path/that/does/not/exist")
        assert result == []

    async def test_write_and_read_file(self, tmp_path) -> None:
        s = Settings()
        client = LLMClient(settings=s)
        filepath = tmp_path / "test.txt"
        await client.write_file(filepath, "hello world")
        content = await client.read_file(filepath)
        assert content == "hello world"

    async def test_write_file_creates_parent_dirs(self, tmp_path) -> None:
        s = Settings()
        client = LLMClient(settings=s)
        filepath = tmp_path / "sub" / "dir" / "test.txt"
        await client.write_file(filepath, "nested content")
        assert filepath.exists()
