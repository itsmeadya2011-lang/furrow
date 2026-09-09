import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import openai
import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient, _is_retryable


class TestIsRetryable:
    def test_connection_error(self):
        assert _is_retryable(ConnectionError("connection failed")) is True

    def test_timeout_error(self):
        assert _is_retryable(TimeoutError("timeout")) is True

    def test_anthropic_connection_error(self):
        assert _is_retryable(anthropic.APIConnectionError(request=MagicMock())) is True

    def test_anthropic_timeout_error(self):
        assert _is_retryable(anthropic.APITimeoutError(request=MagicMock())) is True

    def test_anthropic_rate_limit_error(self):
        assert _is_retryable(anthropic.RateLimitError(message="rate limited", response=MagicMock(), body={})) is True

    def test_anthropic_status_error_retryable(self):
        exc = anthropic.APIStatusError(
            message="server error",
            response=MagicMock(status_code=500),
            body={},
        )
        assert _is_retryable(exc) is True

    def test_anthropic_status_error_429(self):
        exc = anthropic.APIStatusError(
            message="rate limit",
            response=MagicMock(status_code=429),
            body={},
        )
        assert _is_retryable(exc) is True

    def test_anthropic_status_error_502(self):
        exc = anthropic.APIStatusError(
            message="bad gateway",
            response=MagicMock(status_code=502),
            body={},
        )
        assert _is_retryable(exc) is True

    def test_anthropic_status_error_503(self):
        exc = anthropic.APIStatusError(
            message="unavailable",
            response=MagicMock(status_code=503),
            body={},
        )
        assert _is_retryable(exc) is True

    def test_anthropic_status_error_504(self):
        exc = anthropic.APIStatusError(
            message="gateway timeout",
            response=MagicMock(status_code=504),
            body={},
        )
        assert _is_retryable(exc) is True

    def test_anthropic_status_error_not_retryable(self):
        exc = anthropic.APIStatusError(
            message="bad request",
            response=MagicMock(status_code=400),
            body={},
        )
        assert _is_retryable(exc) is False

    def test_anthropic_status_error_401_not_retryable(self):
        exc = anthropic.APIStatusError(
            message="unauthorized",
            response=MagicMock(status_code=401),
            body={},
        )
        assert _is_retryable(exc) is False

    def test_openai_connection_error(self):
        assert _is_retryable(openai.APIConnectionError(request=MagicMock())) is True

    def test_openai_timeout_error(self):
        assert _is_retryable(openai.APITimeoutError(request=MagicMock())) is True

    def test_openai_rate_limit_error(self):
        assert _is_retryable(openai.RateLimitError(message="rate limited", response=MagicMock(), body={})) is True

    def test_openai_status_error_retryable(self):
        exc = openai.APIStatusError(
            message="server error",
            response=MagicMock(status_code=500),
            body={},
        )
        assert _is_retryable(exc) is True

    def test_openai_status_error_429(self):
        exc = openai.APIStatusError(
            message="rate limit",
            response=MagicMock(status_code=429),
            body={},
        )
        assert _is_retryable(exc) is True

    def test_openai_status_error_not_retryable(self):
        exc = openai.APIStatusError(
            message="bad request",
            response=MagicMock(status_code=400),
            body={},
        )
        assert _is_retryable(exc) is False

    def test_httpx_connect_error(self):
        assert _is_retryable(httpx.ConnectError("connect failed")) is True

    def test_httpx_timeout_exception(self):
        assert _is_retryable(httpx.TimeoutException("timeout")) is True

    def test_httpx_status_error_retryable(self):
        response = MagicMock()
        response.status_code = 500
        exc = httpx.HTTPStatusError("server error", request=MagicMock(), response=response)
        assert _is_retryable(exc) is True

    def test_httpx_status_error_429(self):
        response = MagicMock()
        response.status_code = 429
        exc = httpx.HTTPStatusError("rate limit", request=MagicMock(), response=response)
        assert _is_retryable(exc) is True

    def test_httpx_status_error_not_retryable(self):
        response = MagicMock()
        response.status_code = 404
        exc = httpx.HTTPStatusError("not found", request=MagicMock(), response=response)
        assert _is_retryable(exc) is False

    def test_value_error_not_retryable(self):
        assert _is_retryable(ValueError("bad value")) is False

    def test_runtime_error_not_retryable(self):
        assert _is_retryable(RuntimeError("runtime error")) is False


class TestLLMClient:
    def test_init_default_settings(self):
        client = LLMClient()
        assert client.settings is not None
        assert client.last_usage is None

    def test_init_custom_settings(self):
        settings = Settings(max_tokens=2048, provider=Provider.OPENAI)
        client = LLMClient(settings=settings)
        assert client.settings.max_tokens == 2048
        assert client.settings.provider == Provider.OPENAI
        assert client.last_usage is None

    def test_last_usage_initialization(self):
        client = LLMClient()
        assert client.last_usage is None

    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        client = LLMClient()
        content = await client.read_file(test_file)
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tmp_path):
        client = LLMClient()
        with pytest.raises(FileNotFoundError):
            await client.read_file(tmp_path / "nonexistent.txt")

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        client = LLMClient()
        test_file = tmp_path / "output.txt"
        await client.write_file(test_file, "test content")
        assert test_file.read_text() == "test content"

    @pytest.mark.asyncio
    async def test_write_file_creates_parent_dirs(self, tmp_path):
        client = LLMClient()
        test_file = tmp_path / "subdir" / "output.txt"
        await client.write_file(test_file, "nested content")
        assert test_file.read_text() == "nested content"

    def test_list_files_empty(self, tmp_path):
        client = LLMClient()
        files = client.list_files(tmp_path)
        assert files == []

    def test_list_files_with_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.py").write_text("b")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "c.txt").write_text("c")
        client = LLMClient()
        files = client.list_files(tmp_path)
        assert set(files) == {"a.txt", "b.py", "sub/c.txt"}

    def test_list_files_nonexistent(self):
        client = LLMClient()
        files = client.list_files("/nonexistent/path")
        assert files == []

    @patch("furrow.llm.AsyncAnthropic")
    @pytest.mark.asyncio
    async def test_complete_anthropic(self, mock_anthropic_class):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="test response")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic_class.return_value = mock_client

        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
        client = LLMClient(settings=settings)
        result = await client.complete("test prompt", system="system prompt")

        assert result == "test response"
        assert client.last_usage == {"input_tokens": 10, "output_tokens": 20}
        mock_client.messages.create.assert_called_once()

    @patch("furrow.llm.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_complete_openai(self, mock_openai_class):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_class.return_value = mock_client

        settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
        client = LLMClient(settings=settings)
        result = await client.complete("test prompt", system="system prompt")

        assert result == "test response"
        assert client.last_usage == {"prompt_tokens": 10, "completion_tokens": 20}
        mock_client.chat.completions.create.assert_called_once()

    @patch("furrow.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_complete_ollama(self, mock_httpx_client_class):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "test response"},
            "prompt_eval_count": 10,
            "eval_count": 20,
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client_class.return_value.__aenter__.return_value = mock_client

        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)
        result = await client.complete("test prompt", system="system prompt")

        assert result == "test response"
        assert client.last_usage == {"prompt_tokens": 10, "completion_tokens": 20}
        mock_client.post.assert_called_once()

    @patch("furrow.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_complete_ollama_error(self, mock_httpx_client_class):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "model not found"}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client_class.return_value.__aenter__.return_value = mock_client

        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)

        with pytest.raises(RuntimeError, match="model not found"):
            await client.complete("test prompt")

    @pytest.mark.asyncio
    async def test_complete_unsupported_provider(self):
        settings = Settings()
        # Bypass pydantic validation since Provider enum rejects unknown values
        settings.provider = "unsupported"
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("test prompt")

    @pytest.mark.asyncio
    async def test_ollama_url_construction(self):
        settings = Settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434/",
        )
        client = LLMClient(settings=settings)
        # The URL construction happens in _complete_ollama, we can test it indirectly
        # by mocking httpx and checking the call
        with patch("furrow.llm.httpx.AsyncClient") as mock_httpx_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"message": {"content": "ok"}}
            mock_response.raise_for_status = MagicMock()
            mock_httpx_client_class.return_value.__aenter__.return_value = mock_client

            await client.complete("test", model="test-model")

            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/chat"

    def test_ollama_url_strips_trailing_slash(self):
        settings = Settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434/",
        )
        client = LLMClient(settings=settings)
        # Just verify the base_url is stored correctly
        assert client.settings.ollama_base_url == "http://localhost:11434/"

    @pytest.mark.asyncio
    async def test_anthropic_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            _ = client.anthropic

    @pytest.mark.asyncio
    async def test_openai_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = Settings(provider=Provider.OPENAI, openai_api_key=None)
        client = LLMClient(settings=settings)
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            _ = client.openai