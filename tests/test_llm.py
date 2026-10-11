import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from furrow.llm import LLMClient
from furrow.config import Settings, Provider


class TestLLMClient:
    @pytest.fixture
    def settings(self):
        return Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
            openai_api_key="test-key",
            ollama_base_url="http://localhost:11434",
        )

    @pytest.fixture
    def client(self, settings):
        return LLMClient(settings=settings)

    def test_client_initializes_with_anthropic(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
        client = LLMClient(settings=settings)
        assert client.settings.provider == Provider.ANTHROPIC

    def test_client_initializes_with_openai(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
        client = LLMClient(settings=settings)
        assert client.settings.provider == Provider.OPENAI

    def test_client_initializes_with_ollama(self):
        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://test:11434")
        client = LLMClient(settings=settings)
        assert client.settings.provider == Provider.OLLAMA

    def test_client_raises_without_anthropic_key(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
        client = LLMClient(settings=settings)
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
                _ = client.anthropic

    def test_client_raises_without_openai_key(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key=None)
        client = LLMClient(settings=settings)
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
                _ = client.openai

    @pytest.mark.asyncio
    async def test_complete_routes_to_anthropic(self, client):
        with patch.object(client, '_complete_anthropic', new_callable=AsyncMock) as mock:
            mock.return_value = "anthropic response"
            client.settings.provider = Provider.ANTHROPIC
            result = await client.complete("test prompt")
            assert result == "anthropic response"
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_routes_to_openai(self, client):
        with patch.object(client, '_complete_openai', new_callable=AsyncMock) as mock:
            mock.return_value = "openai response"
            client.settings.provider = Provider.OPENAI
            result = await client.complete("test prompt")
            assert result == "openai response"
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_routes_to_ollama(self, client):
        with patch.object(client, '_complete_ollama', new_callable=AsyncMock) as mock:
            mock.return_value = "ollama response"
            client.settings.provider = Provider.OLLAMA
            result = await client.complete("test prompt")
            assert result == "ollama response"
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_raises_for_unknown_provider(self, client):
        client.settings.provider = "unknown"
        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("test prompt")

    @pytest.mark.asyncio
    async def test_read_file(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("hello world")

            result = await client.read_file(test_file)
            assert result == "hello world"

    @pytest.mark.asyncio
    async def test_write_file(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "subdir" / "test.txt"
            content = "new content"

            await client.write_file(test_file, content)
            assert test_file.exists()
            assert test_file.read_text() == content

    def test_list_files(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "file1.txt").write_text("1")
            (test_dir / "file2.txt").write_text("2")
            subdir = test_dir / "subdir"
            subdir.mkdir()
            (subdir / "file3.txt").write_text("3")

            files = client.list_files(test_dir)
            assert len(files) == 3
            assert "file1.txt" in files
            assert "file2.txt" in files
            assert "subdir/file3.txt" in files

    def test_list_files_nonexistent_directory(self, client):
        files = client.list_files("/nonexistent/path")
        assert files == []

    @pytest.mark.asyncio
    async def test_complete_anthropic(self, client):
        with patch('furrow.llm.AsyncAnthropic') as mock_anthropic:
            mock_instance = MagicMock()
            mock_instance.messages.create = AsyncMock(return_value=MagicMock(
                content=[MagicMock(text="response")]
            ))
            mock_anthropic.return_value = mock_instance

            client.settings.provider = Provider.ANTHROPIC
            result = await client._complete_anthropic("prompt", "system", "model")
            assert result == "response"

    @pytest.mark.asyncio
    async def test_complete_openai(self, client):
        with patch('furrow.llm.AsyncOpenAI') as mock_openai:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="response"))]
            ))
            mock_openai.return_value = mock_instance

            client.settings.provider = Provider.OPENAI
            result = await client._complete_openai("prompt", "system", "model")
            assert result == "response"

    @pytest.mark.asyncio
    async def test_complete_ollama(self, client):
        with patch('furrow.llm.AsyncOpenAI') as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="response"))]
            ))
            mock_ollama.return_value = mock_instance

            client.settings.provider = Provider.OLLAMA
            result = await client._complete_ollama("prompt", "system", "model")
            assert result == "response"