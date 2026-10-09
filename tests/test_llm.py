from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


class TestLLMClient:
    @pytest.fixture
    def settings(self):
        return Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key=None,
            openai_api_key=None,
            ollama_base_url="http://localhost:11434"
        )

    def test_anthropic_raises_value_error_when_no_api_key(self, settings):
        settings.provider = Provider.ANTHROPIC
        settings.anthropic_api_key = None

        with patch.dict('os.environ', {}, clear=True):
            client = LLMClient(settings=settings)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
                _ = client.anthropic

    def test_anthropic_uses_env_var_when_available(self, settings):
        settings.provider = Provider.ANTHROPIC
        settings.anthropic_api_key = None

        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'env-key-123'}, clear=True):
            client = LLMClient(settings=settings)
            with patch('furrow.llm.AsyncAnthropic') as mock_anthropic:
                _ = client.anthropic
                mock_anthropic.assert_called_once_with(api_key='env-key-123')

    def test_anthropic_uses_settings_key_when_provided(self, settings):
        settings.provider = Provider.ANTHROPIC
        settings.anthropic_api_key = "settings-key-456"

        with patch.dict('os.environ', {}, clear=True):
            client = LLMClient(settings=settings)
            with patch('furrow.llm.AsyncAnthropic') as mock_anthropic:
                _ = client.anthropic
                mock_anthropic.assert_called_once_with(api_key='settings-key-456')

    def test_openai_raises_value_error_when_no_api_key(self, settings):
        settings.provider = Provider.OPENAI
        settings.openai_api_key = None

        with patch.dict('os.environ', {}, clear=True):
            client = LLMClient(settings=settings)
            with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
                _ = client.openai

    def test_ollama_client_created_lazily_with_correct_base_url(self, settings):
        settings.provider = Provider.OLLAMA
        settings.ollama_base_url = "http://custom-host:9999"

        client = LLMClient(settings=settings)

        assert client._ollama is None
        _ = client.ollama
        assert client._ollama is not None
        assert str(client._ollama.base_url) == "http://custom-host:9999"

    def test_ollama_client_reuses_existing_instance(self, settings):
        settings.provider = Provider.OLLAMA

        client = LLMClient(settings=settings)
        ollama1 = client.ollama
        ollama2 = client.ollama

        assert ollama1 is ollama2

    def test_complete_dispatches_to_anthropic_when_provider_anthropic(self, settings):
        settings.provider = Provider.ANTHROPIC
        settings.anthropic_api_key = "test-key"

        client = LLMClient(settings=settings)

        with patch.object(client, '_complete_anthropic', new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "Anthropic response"
            import asyncio
            result = asyncio.run(client.complete("Test prompt"))

        assert result == "Anthropic response"
        mock_complete.assert_called_once_with("Test prompt", "", settings.model)

    def test_complete_dispatches_to_openai_when_provider_openai(self, settings):
        settings.provider = Provider.OPENAI
        settings.openai_api_key = "test-key"

        client = LLMClient(settings=settings)

        with patch.object(client, '_complete_openai', new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "OpenAI response"
            import asyncio
            result = asyncio.run(client.complete("Test prompt"))

        assert result == "OpenAI response"
        mock_complete.assert_called_once_with("Test prompt", "", settings.model)

    def test_complete_dispatches_to_ollama_when_provider_ollama(self, settings):
        settings.provider = Provider.OLLAMA

        client = LLMClient(settings=settings)

        with patch.object(client, '_complete_ollama', new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "Ollama response"
            import asyncio
            result = asyncio.run(client.complete("Test prompt"))

        assert result == "Ollama response"
        mock_complete.assert_called_once_with("Test prompt", "", settings.model)

    def test_complete_raises_value_error_for_unsupported_provider(self, settings):
        settings.provider = "unsupported"

        client = LLMClient(settings=settings)

        import asyncio
        with pytest.raises(ValueError, match="Unsupported provider"):
            asyncio.run(client.complete("Test prompt"))

    def test_complete_passes_model_when_provided(self, settings):
        settings.provider = Provider.ANTHROPIC
        settings.anthropic_api_key = "test-key"

        client = LLMClient(settings=settings)

        with patch.object(client, '_complete_anthropic', new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "Response"
            import asyncio
            asyncio.run(client.complete("Test prompt", model="custom-model"))

        mock_complete.assert_called_once_with("Test prompt", "", "custom-model")

    def test_complete_passes_system_prompt(self, settings):
        settings.provider = Provider.ANTHROPIC
        settings.anthropic_api_key = "test-key"

        client = LLMClient(settings=settings)

        with patch.object(client, '_complete_anthropic', new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "Response"
            import asyncio
            asyncio.run(client.complete("Test prompt", system="System prompt"))

        mock_complete.assert_called_once_with("Test prompt", "System prompt", settings.model)

    @pytest.mark.asyncio
    async def test_read_file_reads_content(self, settings):
        client = LLMClient(settings=settings)

        with patch('aiofiles.open') as mock_open:
            mock_file = AsyncMock()
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock(return_value=None)
            mock_file.read = AsyncMock(return_value="file content")
            mock_open.return_value = mock_file

            content = await client.read_file("/path/to/file.txt")

        assert content == "file content"

    @pytest.mark.asyncio
    async def test_write_file_writes_content(self, settings):
        client = LLMClient(settings=settings)

        with patch('aiofiles.open') as mock_open:
            mock_file = AsyncMock()
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock(return_value=None)
            mock_file.write = AsyncMock()
            mock_open.return_value = mock_file

            with patch('pathlib.Path.mkdir'):
                await client.write_file("/path/to/file.txt", "content to write")

        mock_file.write.assert_called_once_with("content to write")

    def test_list_files_returns_relative_paths(self, settings):
        client = LLMClient(settings=settings)

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.rglob') as mock_rglob:
            mock_file1 = MagicMock()
            mock_file1.is_file.return_value = True
            mock_file1.relative_to.return_value = "dir/file1.py"
            mock_file2 = MagicMock()
            mock_file2.is_file.return_value = True
            mock_file2.relative_to.return_value = "dir/file2.py"
            mock_dir = MagicMock()
            mock_dir.is_file.return_value = False
            mock_rglob.return_value = [mock_file1, mock_dir, mock_file2]

            files = client.list_files("/workspace")

        assert files == ["dir/file1.py", "dir/file2.py"]

    def test_list_files_returns_empty_when_directory_not_exists(self, settings):
        client = LLMClient(settings=settings)

        with patch('pathlib.Path.exists', return_value=False):
            files = client.list_files("/nonexistent")

        assert files == []


class TestLLMClientProviderProperties:
    @pytest.fixture
    def settings(self):
        return Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
            openai_api_key="test-key",
            ollama_base_url="http://localhost:11434"
        )

    def test_anthropic_property_caches_client(self, settings):
        client = LLMClient(settings=settings)

        with patch('furrow.llm.AsyncAnthropic') as mock_anthropic:
            mock_instance = MagicMock()
            mock_anthropic.return_value = mock_instance

            anthropic1 = client.anthropic
            anthropic2 = client.anthropic

        assert anthropic1 is anthropic2
        assert anthropic1 is mock_instance
        mock_anthropic.assert_called_once()

    def test_openai_property_caches_client(self, settings):
        client = LLMClient(settings=settings)

        with patch('furrow.llm.AsyncOpenAI') as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            openai1 = client.openai
            openai2 = client.openai

        assert openai1 is openai2
        assert openai1 is mock_instance
        mock_openai.assert_called_once()