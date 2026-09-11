import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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

    def test_retry_decorator_imported(self):
        """Test that tenacity retry decorator is imported and available."""
        import tenacity
        assert hasattr(tenacity, "retry")
        assert hasattr(tenacity, "stop_after_attempt")
        assert hasattr(tenacity, "wait_exponential")
        assert hasattr(tenacity, "retry_if_exception_type")

    def test_llm_client_has_retry_on_complete_methods(self, client):
        """Test that the internal complete methods have retry decorator applied."""
        import tenacity
        
        assert hasattr(client._complete_anthropic, "retry")
        assert hasattr(client._complete_openai, "retry")

    @pytest.mark.asyncio
    async def test_ollama_provider_routed_correctly(self, settings):
        """Test that Ollama provider routes to _complete_ollama."""
        settings.provider = Provider.OLLAMA
        client = LLMClient(settings=settings)
        
        with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock_ollama:
            mock_ollama.return_value = "Ollama response"
            
            result = await client.complete("test prompt")
            
            assert result == "Ollama response"
            mock_ollama.assert_called_once_with("test prompt", "", settings.model)

    @pytest.mark.asyncio
    async def test_anthropic_provider_routed_correctly(self, settings):
        """Test that Anthropic provider routes to _complete_anthropic."""
        settings.provider = Provider.ANTHROPIC
        client = LLMClient(settings=settings)
        
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock_anthropic:
            mock_anthropic.return_value = "Anthropic response"
            
            result = await client.complete("test prompt")
            
            assert result == "Anthropic response"
            mock_anthropic.assert_called_once_with("test prompt", "", settings.model)

    @pytest.mark.asyncio
    async def test_openai_provider_routed_correctly(self, settings):
        """Test that OpenAI provider routes to _complete_openai."""
        settings.provider = Provider.OPENAI
        client = LLMClient(settings=settings)
        
        with patch.object(client, "_complete_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "OpenAI response"
            
            result = await client.complete("test prompt")
            
            assert result == "OpenAI response"
            mock_openai.assert_called_once_with("test prompt", "", settings.model)

    @pytest.mark.asyncio
    async def test_complete_ollama_makes_http_call(self, settings):
        """Test that _complete_ollama makes an HTTP call to Ollama API."""
        settings.provider = Provider.OLLAMA
        client = LLMClient(settings=settings)
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"response": "Ollama result"})
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await client._complete_ollama("test prompt", "system prompt", "llama3")
            
            assert result == "Ollama result"
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == f"{settings.ollama_base_url}/api/generate"
            assert call_args[1]["json"]["model"] == "llama3"
            assert call_args[1]["json"]["prompt"] == "test prompt"
            assert call_args[1]["json"]["system"] == "system prompt"
            assert call_args[1]["json"]["stream"] is False

    @pytest.mark.asyncio
    async def test_complete_ollama_parses_response(self, settings):
        """Test that _complete_ollama parses the response correctly."""
        settings.provider = Provider.OLLAMA
        client = LLMClient(settings=settings)
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"response": "parsed result"})
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await client._complete_ollama("prompt", "", "model")
            
            assert result == "parsed result"

    @pytest.mark.asyncio
    async def test_complete_ollama_raises_on_missing_response_key(self, settings):
        """Test that _complete_ollama raises ValueError when response key missing."""
        settings.provider = Provider.OLLAMA
        client = LLMClient(settings=settings)
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"wrong_key": "value"})
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            with pytest.raises(ValueError, match="Failed to parse Ollama response"):
                await client._complete_ollama("prompt", "", "model")

    def test_read_file(self, client, tmp_path):
        """Test reading a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("file content")
        
        import asyncio
        result = asyncio.run(client.read_file(test_file))
        
        assert result == "file content"

    def test_write_file(self, client, tmp_path):
        """Test writing a file."""
        test_file = tmp_path / "output.txt"
        
        import asyncio
        asyncio.run(client.write_file(test_file, "written content"))
        
        assert test_file.read_text() == "written content"

    def test_write_file_creates_parent_dirs(self, client, tmp_path):
        """Test that write_file creates parent directories."""
        test_file = tmp_path / "subdir" / "output.txt"
        
        import asyncio
        asyncio.run(client.write_file(test_file, "content"))
        
        assert test_file.read_text() == "content"

    def test_list_files(self, client, tmp_path):
        """Test listing files in a directory."""
        (tmp_path / "file1.txt").write_text("1")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file2.txt").write_text("2")
        
        result = client.list_files(tmp_path)
        
        assert set(result) == {"file1.txt", "subdir/file2.txt"}

    def test_list_files_returns_empty_for_nonexistent(self, client, tmp_path):
        """Test listing files for non-existent directory returns empty list."""
        nonexistent = tmp_path / "nonexistent"
        
        result = client.list_files(nonexistent)
        
        assert result == []

    def test_llm_client_init_with_default_settings(self):
        """Test LLMClient can be initialized with default settings."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
            client = LLMClient()
            assert client.settings.provider == Provider.ANTHROPIC