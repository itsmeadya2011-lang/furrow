"""Tests for the LLM client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def openai_settings():
    return Settings(provider=Provider.OPENAI, openai_api_key="test-key")


@pytest.fixture
def anthropic_settings():
    return Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")


@pytest.fixture
def ollama_settings():
    return Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")


class TestLLMClientOpenAI:
    @pytest.mark.asyncio
    async def test_complete_openai(self, openai_settings):
        client = LLMClient(settings=openai_settings)
        # Set the private attribute directly since openai is a read-only property
        mock_openai_client = MagicMock()
        client._openai = mock_openai_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello from OpenAI"
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.complete("Hi", model="gpt-4")
        assert result == "Hello from OpenAI"

    @pytest.mark.asyncio
    async def test_complete_openai_uses_default_model(self, openai_settings):
        client = LLMClient(settings=openai_settings)
        mock_openai_client = MagicMock()
        client._openai = mock_openai_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "result"
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await client.complete("test")
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == openai_settings.model


class TestLLMClientAnthropic:
    @pytest.mark.asyncio
    async def test_complete_anthropic(self, anthropic_settings):
        client = LLMClient(settings=anthropic_settings)
        mock_anthropic_client = MagicMock()
        client._anthropic = mock_anthropic_client
        mock_response = MagicMock()
        mock_response.content[0].text = "Hello from Anthropic"
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        result = await client.complete("Hi", model="claude-3-5-sonnet")
        assert result == "Hello from Anthropic"

    @pytest.mark.asyncio
    async def test_complete_anthropic_uses_default_model(self, anthropic_settings):
        client = LLMClient(settings=anthropic_settings)
        mock_anthropic_client = MagicMock()
        client._anthropic = mock_anthropic_client
        mock_response = MagicMock()
        mock_response.content[0].text = "result"
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        await client.complete("test")
        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == anthropic_settings.model


class TestLLMClientOllama:
    @pytest.mark.asyncio
    async def test_complete_ollama(self, ollama_settings):
        client = LLMClient(settings=ollama_settings)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"response": "Hello from Ollama"})

        mock_http = MagicMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._http_client = mock_http

        result = await client._complete_ollama("Hi", "", "llama3")
        assert result == "Hello from Ollama"

    def test_ollama_base_url(self, ollama_settings):
        client = LLMClient(settings=ollama_settings)
        assert client.settings.ollama_base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_ollama_complete_dispatches(self, ollama_settings):
        client = LLMClient(settings=ollama_settings)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"response": "ollama via complete"})
        mock_http = MagicMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._http_client = mock_http

        result = await client.complete("Hi", model="llama3")
        assert result == "ollama via complete"


class TestLLMClientProviderDispatch:
    @pytest.mark.asyncio
    async def test_provider_dispatch(self, openai_settings):
        """Test that complete() dispatches to the right provider method."""
        client = LLMClient(settings=openai_settings)

        with patch.object(
            client, "_complete_anthropic", new=AsyncMock(return_value="anthropic result")
        ):
            client.settings.provider = Provider.ANTHROPIC
            result = await client.complete("test")
            assert result == "anthropic result"

        with patch.object(client, "_complete_openai", new=AsyncMock(return_value="openai result")):
            client.settings.provider = Provider.OPENAI
            result = await client.complete("test")
            assert result == "openai result"

        with patch.object(client, "_complete_ollama", new=AsyncMock(return_value="ollama result")):
            client.settings.provider = Provider.OLLAMA
            result = await client.complete("test")
            assert result == "ollama result"


class TestLLMClientFileOps:
    @pytest.mark.asyncio
    async def test_write_file_creates_parents(self, tmp_path):
        client = LLMClient(settings=Settings())
        target = tmp_path / "subdir" / "output.txt"
        await client.write_file(target, "hello world")
        assert target.exists()
        assert target.read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        client = LLMClient(settings=Settings())
        test_file = tmp_path / "test.txt"
        test_file.write_text("file contents")
        result = await client.read_file(test_file)
        assert result == "file contents"

    def test_list_files(self, tmp_path):
        client = LLMClient(settings=Settings())
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("y")
        files = client.list_files(tmp_path)
        assert "a.py" in files
        assert "sub/b.py" in files

    def test_list_files_nonexistent(self):
        client = LLMClient(settings=Settings())
        assert client.list_files("/nonexistent/path") == []


class TestLLMClientCleanup:
    @pytest.mark.asyncio
    async def test_aclose(self, ollama_settings):
        client = LLMClient(settings=ollama_settings)
        # Create the http client by accessing it
        _ = client.http_client
        assert client._http_client is not None
        await client.aclose()
        assert client._http_client is None
