from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def _anthropic_client_with_key() -> LLMClient:
    return LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-test"))


def _openai_client_with_key() -> LLMClient:
    return LLMClient(settings=Settings(provider=Provider.OPENAI, openai_api_key="sk-test"))


def _ollama_client() -> LLMClient:
    return LLMClient(settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434"))


class TestCompleteAnthropic:
    async def test_delegates_to_anthropic(self):
        client = _anthropic_client_with_key()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="hello anthropic")]
        mock_anthropic = MagicMock(spec=AsyncAnthropic)
        mock_anthropic.messages.create = AsyncMock(return_value=mock_resp)
        client._anthropic = mock_anthropic

        result = await client.complete("hi", system="sys", model="claude-x")
        assert result == "hello anthropic"
        client._anthropic.messages.create.assert_awaited_once()
        kwargs = client._anthropic.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-x"
        assert kwargs["system"] == "sys"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]

    async def test_default_model_used(self):
        client = _anthropic_client_with_key()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="ok")]
        client._anthropic = MagicMock(spec=AsyncAnthropic)
        client._anthropic.messages.create = AsyncMock(return_value=mock_resp)
        await client.complete("hi")
        kwargs = client._anthropic.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-20250514"

    async def test_default_system_when_empty(self):
        client = _anthropic_client_with_key()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="ok")]
        client._anthropic = MagicMock(spec=AsyncAnthropic)
        client._anthropic.messages.create = AsyncMock(return_value=mock_resp)
        await client.complete("hi", "", "m")
        kwargs = client._anthropic.messages.create.call_args.kwargs
        assert kwargs["system"] == "You are a helpful coding assistant."


class TestCompleteOpenAI:
    async def test_delegates_to_openai(self):
        client = _openai_client_with_key()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="hello openai"))]
        client._openai = MagicMock(spec=AsyncOpenAI)
        client._openai.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.complete("hi", system="sys", model="gpt-4")
        assert result == "hello openai"
        client._openai.chat.completions.create.assert_awaited_once()
        kwargs = client._openai.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-4"

    async def test_none_content_returns_empty_string(self):
        client = _openai_client_with_key()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=None))]
        client._openai = MagicMock(spec=AsyncOpenAI)
        client._openai.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await client.complete("hi")
        assert result == ""


class TestOllama:
    async def test_ollama_completion(self):
        client = _ollama_client()
        mock_http = MagicMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"message": {"content": "hello ollama"}})
        mock_http.post = AsyncMock(return_value=mock_response)
        client._http = mock_http

        result = await client.complete("hi", system="sys", model="llama")
        assert result == "hello ollama"
        client._http.post.assert_awaited_once()
        kwargs = client._http.post.call_args.kwargs
        assert kwargs["json"]["model"] == "llama"
        assert kwargs["json"]["messages"][0]["role"] == "system"
        assert kwargs["json"]["messages"][1]["role"] == "user"


class TestRetry:
    async def test_retries_then_succeeds(self):
        client = _anthropic_client_with_key()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="ok")]
        client._anthropic = MagicMock(spec=AsyncAnthropic)
        side = [RuntimeError("boom"), mock_resp]
        client._anthropic.messages.create = AsyncMock(side_effect=side)

        result = await client.complete("hi")
        assert result == "ok"
        assert client._anthropic.messages.create.await_count == 2

    async def test_raises_after_max_retries(self):
        client = _anthropic_client_with_key()
        client._anthropic = MagicMock(spec=AsyncAnthropic)
        client._anthropic.messages.create = AsyncMock(side_effect=RuntimeError("always"))

        with pytest.raises(RuntimeError, match="always"):
            await client.complete("hi")


class TestFileOperations:
    async def test_write_and_read(self, tmp_path):
        client = _ollama_client()
        f = tmp_path / "sub" / "file.txt"
        await client.write_file(f, "content")
        assert f.read_text() == "content"
        assert await client.read_file(f) == "content"

    async def test_edit_file(self, tmp_path):
        client = _ollama_client()
        f = tmp_path / "file.txt"
        f.write_text("hello world")
        await client.edit_file(f, "hello", "goodbye")
        assert f.read_text() == "goodbye world"

    async def test_edit_file_not_found_raises(self, tmp_path):
        client = _ollama_client()
        f = tmp_path / "file.txt"
        f.write_text("only this")
        with pytest.raises(ValueError, match="String not found"):
            await client.edit_file(f, "missing", "x")

    async def test_delete_file(self, tmp_path):
        client = _ollama_client()
        f = tmp_path / "file.txt"
        f.write_text("bye")
        await client.delete_file(f)
        assert not f.exists()

    async def test_delete_missing_file_no_error(self, tmp_path):
        client = _ollama_client()
        f = tmp_path / "gone.txt"
        await client.delete_file(f)
        assert not f.exists()

    def test_list_files(self, tmp_path):
        client = _ollama_client()
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("y")
        files = client.list_files(tmp_path)
        rel = {str(Path(f)) for f in files}
        assert "a.py" in rel
        assert str(Path("sub") / "b.py") in rel

    def test_list_files_missing_dir(self, tmp_path):
        client = _ollama_client()
        assert client.list_files(tmp_path / "nope") == []
