import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from furrow.llm import LLMClient
from furrow.config import Settings, Provider


class TestLLMClient:
    def test_client_initialization(self):
        settings = Settings()
        client = LLMClient(settings=settings)
        assert client.settings == settings
        assert client._anthropic is None
        assert client._openai is None
        assert client._ollama_client is None

    @pytest.mark.asyncio
    async def test_anthropic_property_creates_client(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
        client = LLMClient(settings=settings)
        with patch("furrow.llm.AsyncAnthropic") as mock_anthropic:
            mock_instance = AsyncMock()
            mock_anthropic.return_value = mock_instance
            anthropic_client = client.anthropic
            assert anthropic_client == mock_instance
            mock_anthropic.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_openai_property_creates_client(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
        client = LLMClient(settings=settings)
        with patch("furrow.llm.AsyncOpenAI") as mock_openai:
            mock_instance = AsyncMock()
            mock_openai.return_value = mock_instance
            openai_client = client.openai
            assert openai_client == mock_instance
            mock_openai.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_ollama_property_creates_client(self):
        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)
        ollama_client = client.ollama
        assert ollama_client is not None
        assert str(ollama_client.base_url).rstrip("/") == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_complete_anthropic(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
        client = LLMClient(settings=settings)
        # Directly set the private attribute to mock it
        mock_anthropic = AsyncMock()
        client._anthropic = mock_anthropic
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        result = await client.complete("Test prompt", model="test-model")
        assert result == "Test response"
        mock_anthropic.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_openai(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
        client = LLMClient(settings=settings)
        mock_openai = AsyncMock()
        client._openai = mock_openai
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await client.complete("Test prompt", model="test-model")
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_complete_ollama(self):
        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)
        mock_ollama = AsyncMock()
        client._ollama_client = mock_ollama
        mock_response = AsyncMock()
        mock_response.json = MagicMock(return_value={"message": {"content": "Test response"}})
        mock_response.raise_for_status = MagicMock()
        mock_ollama.post = AsyncMock(return_value=mock_response)
        result = await client.complete("Test prompt", model="llama3.1")
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_read_write_file(self, tmp_path):
        settings = Settings(workspace=tmp_path)
        client = LLMClient(settings=settings)
        test_file = tmp_path / "test.txt"
        await client.write_file(test_file, "Hello World")
        content = await client.read_file(test_file)
        assert content == "Hello World"

    @pytest.mark.asyncio
    async def test_list_files(self, tmp_path):
        settings = Settings(workspace=tmp_path)
        client = LLMClient(settings=settings)
        (tmp_path / "file1.txt").write_text("1")
        (tmp_path / "dir").mkdir()
        (tmp_path / "dir" / "file2.txt").write_text("2")
        files = client.list_files(tmp_path)
        assert "file1.txt" in files
        assert "dir/file2.txt" in files

    @pytest.mark.asyncio
    async def test_close_clients(self):
        settings = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=settings)
        mock_ollama = AsyncMock()
        client._ollama_client = mock_ollama
        await client.close()
        mock_ollama.aclose.assert_called_once()