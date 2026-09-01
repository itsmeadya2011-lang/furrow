from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


class TestLLMClient:
    def test_init_default_settings(self):
        client = LLMClient()
        assert client._anthropic is None
        assert client._openai is None
        assert client._ollama is None

    def test_init_custom_settings(self, test_settings):
        client = LLMClient(settings=test_settings)
        assert client.settings == test_settings

    @pytest.mark.asyncio
    async def test_complete_anthropic(self, test_settings):
        test_settings.provider = Provider.ANTHROPIC
        client = LLMClient(settings=test_settings)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="anthropic response")]

        mock_messages = AsyncMock()
        mock_messages.create = AsyncMock(return_value=mock_response)

        mock_anthropic = MagicMock()
        mock_anthropic.messages = mock_messages

        with patch.object(client, "_anthropic", mock_anthropic):
            result = await client.complete("test prompt", system="test system")
            assert result == "anthropic response"
            mock_messages.create.assert_called_once_with(
                model="test-model",
                max_tokens=4096,
                system="test system",
                messages=[{"role": "user", "content": "test prompt"}],
            )

    @pytest.mark.asyncio
    async def test_complete_openai(self, test_settings):
        test_settings.provider = Provider.OPENAI
        client = LLMClient(settings=test_settings)

        mock_message = MagicMock()
        mock_message.content = "openai response"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_chat = MagicMock()
        mock_chat.completions.create = AsyncMock(return_value=mock_response)

        mock_openai = MagicMock()
        mock_openai.chat = mock_chat

        with patch.object(client, "_openai", mock_openai):
            result = await client.complete("test prompt", system="test system")
            assert result == "openai response"
            mock_chat.completions.create.assert_called_once_with(
                model="test-model",
                messages=[
                    {"role": "system", "content": "test system"},
                    {"role": "user", "content": "test prompt"},
                ],
            )

    @pytest.mark.asyncio
    async def test_complete_ollama(self, test_settings):
        test_settings.provider = Provider.OLLAMA
        client = LLMClient(settings=test_settings)

        mock_message = MagicMock()
        mock_message.content = "ollama response"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_chat = MagicMock()
        mock_chat.completions.create = AsyncMock(return_value=mock_response)

        mock_ollama = MagicMock()
        mock_ollama.chat = mock_chat

        with patch.object(client, "_ollama", mock_ollama):
            result = await client.complete("test prompt", system="test system")
            assert result == "ollama response"
            mock_chat.completions.create.assert_called_once_with(
                model="test-model",
                messages=[
                    {"role": "system", "content": "test system"},
                    {"role": "user", "content": "test prompt"},
                ],
            )

    @pytest.mark.asyncio
    async def test_complete_unsupported_provider(self, test_settings):
        test_settings.provider = "unsupported"
        client = LLMClient(settings=test_settings)

        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("test prompt")

    @pytest.mark.asyncio
    async def test_complete_uses_default_model(self, test_settings):
        test_settings.provider = Provider.ANTHROPIC
        client = LLMClient(settings=test_settings)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]

        mock_messages = AsyncMock()
        mock_messages.create = AsyncMock(return_value=mock_response)

        mock_anthropic = MagicMock()
        mock_anthropic.messages = mock_messages

        with patch.object(client, "_anthropic", mock_anthropic):
            await client.complete("test prompt")
            mock_messages.create.assert_called_once()
            call_kwargs = mock_messages.create.call_args[1]
            assert call_kwargs["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_complete_uses_custom_model(self, test_settings):
        test_settings.provider = Provider.ANTHROPIC
        client = LLMClient(settings=test_settings)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]

        mock_messages = AsyncMock()
        mock_messages.create = AsyncMock(return_value=mock_response)

        mock_anthropic = MagicMock()
        mock_anthropic.messages = mock_messages

        with patch.object(client, "_anthropic", mock_anthropic):
            await client.complete("test prompt", model="custom-model")
            call_kwargs = mock_messages.create.call_args[1]
            assert call_kwargs["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_complete_openai_empty_content(self, test_settings):
        test_settings.provider = Provider.OPENAI
        client = LLMClient(settings=test_settings)

        mock_message = MagicMock()
        mock_message.content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_chat = MagicMock()
        mock_chat.completions.create = AsyncMock(return_value=mock_response)

        mock_openai = MagicMock()
        mock_openai.chat = mock_chat

        with patch.object(client, "_openai", mock_openai):
            result = await client.complete("test prompt")
            assert result == ""

    def test_ollama_property_creates_client_with_base_url(self, test_settings):
        test_settings.provider = Provider.OLLAMA
        client = LLMClient(settings=test_settings)

        with patch("furrow.llm.AsyncOpenAI") as mock_openai_class:
            _ = client.ollama
            mock_openai_class.assert_called_once_with(
                base_url="http://localhost:11434",
                api_key="ollama",
            )

    def test_ollama_property_caches_client(self, test_settings):
        test_settings.provider = Provider.OLLAMA
        client = LLMClient(settings=test_settings)

        with patch("furrow.llm.AsyncOpenAI") as mock_openai_class:
            mock_openai_class.return_value = MagicMock()

            first_call = client.ollama
            second_call = client.ollama

            assert first_call is second_call
            mock_openai_class.assert_called_once()

    def test_anthropic_property_raises_without_api_key(self, test_settings):
        test_settings.anthropic_api_key = None
        client = LLMClient(settings=test_settings)

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
                _ = client.anthropic

    def test_openai_property_raises_without_api_key(self, test_settings):
        test_settings.openai_api_key = None
        client = LLMClient(settings=test_settings)

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
                _ = client.openai

    def test_anthropic_property_uses_env_var(self, test_settings):
        test_settings.anthropic_api_key = None
        client = LLMClient(settings=test_settings)

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            with patch("furrow.llm.AsyncAnthropic") as mock_anthropic_class:
                _ = client.anthropic
                mock_anthropic_class.assert_called_once_with(api_key="env-key")

    def test_openai_property_uses_env_var(self, test_settings):
        test_settings.openai_api_key = None
        client = LLMClient(settings=test_settings)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            with patch("furrow.llm.AsyncOpenAI") as mock_openai_class:
                _ = client.openai
                mock_openai_class.assert_called_once_with(api_key="env-key")

    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        client = LLMClient()
        result = await client.read_file(test_file)
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        test_file = tmp_path / "subdir" / "test.txt"

        client = LLMClient()
        await client.write_file(test_file, "test content")

        assert test_file.exists()
        assert test_file.read_text() == "test content"

    def test_list_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "c.md").write_text("c")

        client = LLMClient()
        files = client.list_files(tmp_path)

        assert "a.txt" in files
        assert "b.py" in files
        assert "subdir/c.md" in files

    def test_list_files_nonexistent_directory(self):
        client = LLMClient()
        files = client.list_files("/nonexistent/path")
        assert files == []
