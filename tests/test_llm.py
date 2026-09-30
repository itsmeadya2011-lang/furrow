from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import wait_fixed

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def fast_retry(monkeypatch):
    """Patch tenacity retry wait to zero so tests run instantaneously."""
    monkeypatch.setattr(LLMClient.complete.retry, "wait", wait_fixed(0))


def _make_anthropic_client():
    """Create an LLMClient with settings that allow instantiating the anthropic property."""
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    return LLMClient(settings=settings)


def _make_openai_client():
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    return LLMClient(settings=settings)


def _make_ollama_client():
    settings = Settings(provider=Provider.OLLAMA, anthropic_api_key="test-key")
    return LLMClient(settings=settings)


class TestLLMClientInstantiation:
    def test_can_be_instantiated_with_default_settings(self):
        client = LLMClient()
        assert client.settings is not None
        assert client._anthropic is None
        assert client._openai is None
        assert client._httpx_client is None

    def test_can_be_instantiated_with_custom_settings(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key="custom-key")
        client = LLMClient(settings=settings)
        assert client.settings is settings
        assert client.settings.provider == Provider.OPENAI
        assert client.settings.openai_api_key == "custom-key"

    def test_lazy_initialization_of_clients(self):
        client = _make_anthropic_client()
        assert client._anthropic is None
        assert client._openai is None


class TestCompleteDispatch:
    async def test_dispatches_to_anthropic(self, fast_retry):
        client = _make_anthropic_client()
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = "anthropic result"
            result = await client.complete("prompt", system="sys", model="test-model")
            mock_method.assert_called_once_with("prompt", "sys", "test-model")
            assert result == "anthropic result"

    async def test_dispatches_to_openai(self, fast_retry):
        client = _make_openai_client()
        with patch.object(client, "_complete_openai", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = "openai result"
            result = await client.complete("prompt", system="sys", model="test-model")
            mock_method.assert_called_once_with("prompt", "sys", "test-model")
            assert result == "openai result"

    async def test_dispatches_to_ollama(self, fast_retry):
        client = _make_ollama_client()
        with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = "ollama result"
            result = await client.complete("prompt", system="sys", model="test-model")
            mock_method.assert_called_once_with("prompt", "sys", "test-model")
            assert result == "ollama result"

    async def test_unsupported_provider_raises(self, fast_retry):
        settings = Settings(provider=Provider.OLLAMA, anthropic_api_key="test-key")
        client = LLMClient(settings=settings)
        client._complete_ollama = AsyncMock(side_effect=ValueError("Unsupported provider"))
        client.settings = Settings(provider=Provider.OPENAI, openai_api_key="test")
        client._complete_openai = AsyncMock(side_effect=ValueError("Unsupported provider"))
        client.settings = Settings(provider=Provider.OLLAMA, anthropic_api_key="test")

        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("prompt", system="sys", model="test-model")


class TestAnthropicParsing:
    async def test_complete_anthropic_parsing(self, fast_retry):
        client = _make_anthropic_client()
        mock_response = MagicMock()
        mock_content_item = MagicMock()
        mock_content_item.text = "anthropic response text"
        mock_response.content = [mock_content_item]

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        client._anthropic = mock_anthropic

        result = await client.complete("prompt", system="sys", model="claude-model")
        assert result == "anthropic response text"
        mock_anthropic.messages.create.assert_called_once_with(
            model="claude-model",
            max_tokens=4096,
            system="sys",
            messages=[{"role": "user", "content": "prompt"}],
        )

    async def test_anthropic_empty_content_raises(self, fast_retry):
        client = _make_anthropic_client()
        mock_response = MagicMock()
        mock_response.content = []
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        client._anthropic = mock_anthropic

        with pytest.raises(ValueError, match="empty or invalid content"):
            await client.complete("prompt", system="sys", model="claude-model")

    async def test_anthropic_no_text_raises(self, fast_retry):
        client = _make_anthropic_client()
        mock_content_item = MagicMock()
        mock_content_item.text = ""
        mock_response = MagicMock()
        mock_response.content = [mock_content_item]
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        client._anthropic = mock_anthropic

        with pytest.raises(ValueError, match="no text content"):
            await client.complete("prompt", system="sys", model="claude-model")

    async def test_anthropic_default_system(self, fast_retry):
        client = _make_anthropic_client()
        mock_response = MagicMock()
        mock_content_item = MagicMock()
        mock_content_item.text = "ok"
        mock_response.content = [mock_content_item]
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        client._anthropic = mock_anthropic

        await client.complete("prompt")
        call_kwargs = mock_anthropic.messages.create.call_args
        assert call_kwargs.kwargs["system"] == "You are a helpful coding assistant."


class TestOpenAIParsing:
    async def test_complete_openai_parsing(self, fast_retry):
        client = _make_openai_client()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "openai response content"

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        client._openai = mock_openai

        result = await client.complete("prompt", system="sys", model="gpt-4")
        assert result == "openai response content"
        mock_openai.chat.completions.create.assert_called_once()

    async def test_openai_no_choices_raises(self, fast_retry):
        client = _make_openai_client()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        client._openai = mock_openai

        with pytest.raises(ValueError, match="no choices"):
            await client.complete("prompt", system="sys", model="gpt-4")

    async def test_openai_empty_content_raises(self, fast_retry):
        client = _make_openai_client()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        client._openai = mock_openai

        with pytest.raises(ValueError, match="empty content"):
            await client.complete("prompt", system="sys", model="gpt-4")

    async def test_openai_default_system(self, fast_retry):
        client = _make_openai_client()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        client._openai = mock_openai

        await client.complete("prompt")
        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful coding assistant."


class TestOllamaParsing:
    async def test_complete_ollama_parsing(self, fast_retry):
        client = _make_ollama_client()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ollama generated text"}
        mock_response.raise_for_status.return_value = None

        mock_http_client = MagicMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch("furrow.llm.httpx.AsyncClient", return_value=mock_http_client):
            result = await client.complete("prompt", system="sys", model="llama3")
            assert result == "ollama generated text"
            mock_http_client.post.assert_called_once()
            call_args = mock_http_client.post.call_args
            assert call_args.args[0] == "http://localhost:11434/api/generate"
            payload = call_args.kwargs["json"]
            assert payload["model"] == "llama3"
            assert payload["prompt"] == "prompt"
            assert payload["system"] == "sys"
            assert payload["stream"] is False

    async def test_ollama_api_error_raises(self, fast_retry):
        client = _make_ollama_client()
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "model not found"}
        mock_response.raise_for_status.return_value = None

        mock_http_client = MagicMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("furrow.llm.httpx.AsyncClient", return_value=mock_http_client),
            pytest.raises(RuntimeError, match="model not found"),
        ):
            await client.complete("prompt", system="sys", model="llama3")

    async def test_ollama_empty_response_raises(self, fast_retry):
        client = _make_ollama_client()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": ""}
        mock_response.raise_for_status.return_value = None

        mock_http_client = MagicMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("furrow.llm.httpx.AsyncClient", return_value=mock_http_client),
            pytest.raises(ValueError, match="empty response"),
        ):
            await client.complete("prompt", system="sys", model="llama3")

    async def test_ollama_default_system(self, fast_retry):
        client = _make_ollama_client()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok"}
        mock_response.raise_for_status.return_value = None

        mock_http_client = MagicMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch("furrow.llm.httpx.AsyncClient", return_value=mock_http_client):
            await client.complete("prompt")
            call_args = mock_http_client.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["system"] == "You are a helpful coding assistant."


class TestRetryBehavior:
    async def test_complete_retries_after_failure(self, fast_retry):
        client = _make_anthropic_client()
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_content_item = MagicMock()
        mock_content_item.text = "success after retry"
        mock_response.content = [mock_content_item]

        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                ConnectionError("network error"),
                mock_response,
            ]
        )
        client._anthropic = mock_anthropic

        result = await client.complete("prompt", system="sys", model="claude")
        assert result == "success after retry"
        assert mock_anthropic.messages.create.call_count == 2

    async def test_complete_all_retries_fail(self, fast_retry):
        client = _make_anthropic_client()
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create = AsyncMock(side_effect=ConnectionError("network error"))
        client._anthropic = mock_anthropic

        with pytest.raises(ConnectionError):
            await client.complete("prompt", system="sys", model="claude")
        assert mock_anthropic.messages.create.call_count == 3

    async def test_complete_succeeds_without_retry(self, fast_retry):
        client = _make_anthropic_client()
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_content_item = MagicMock()
        mock_content_item.text = "immediate success"
        mock_response.content = [mock_content_item]
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        client._anthropic = mock_anthropic

        result = await client.complete("prompt", system="sys", model="claude")
        assert result == "immediate success"
        assert mock_anthropic.messages.create.call_count == 1


class TestFileOperations:
    async def test_read_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        client = LLMClient(settings=Settings(anthropic_api_key="test"))
        result = await client.read_file(test_file)
        assert result == "hello world"

    async def test_read_file_path_str(self, tmp_path):
        test_file = tmp_path / "data.txt"
        test_file.write_text("content here")
        client = LLMClient(settings=Settings(anthropic_api_key="test"))
        result = await client.read_file(str(test_file))
        assert result == "content here"

    async def test_write_file(self, tmp_path):
        test_file = tmp_path / "output.txt"
        client = LLMClient(settings=Settings(anthropic_api_key="test"))
        await client.write_file(test_file, "written content")
        assert test_file.read_text() == "written content"

    async def test_write_file_creates_parent_dirs(self, tmp_path):
        test_file = tmp_path / "subdir1" / "subdir2" / "output.txt"
        client = LLMClient(settings=Settings(anthropic_api_key="test"))
        await client.write_file(test_file, "nested content")
        assert test_file.read_text() == "nested content"

    async def test_write_file_str_path(self, tmp_path):
        test_file = tmp_path / "str_path.txt"
        client = LLMClient(settings=Settings(anthropic_api_key="test"))
        await client.write_file(str(test_file), "str content")
        assert test_file.read_text() == "str content"

    def test_list_files(self, tmp_path):
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.txt").write_text("b")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.txt").write_text("c")

        client = LLMClient(settings=Settings(anthropic_api_key="test"))
        files = client.list_files(tmp_path)
        assert len(files) == 3
        assert "file1.txt" in files
        assert "file2.txt" in files
        assert "file3.txt" not in files
        assert str(Path("subdir") / "file3.txt") in files

    def test_list_files_nonexistent_directory(self, tmp_path):
        client = LLMClient(settings=Settings(anthropic_api_key="test"))
        files = client.list_files(tmp_path / "nonexistent")
        assert files == []

    def test_list_files_empty_directory(self, tmp_path):
        client = LLMClient(settings=Settings(anthropic_api_key="test"))
        files = client.list_files(tmp_path)
        assert files == []
