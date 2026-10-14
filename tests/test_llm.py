from unittest.mock import AsyncMock, MagicMock, patch
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
import pytest
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
from furrow.config import Completion, Provider, Settings
wait client.complete_with_tools("test")
from furrow.llm import LLMClient
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
class TestLLMClientInit:
wait client.complete_with_tools("test")
    def test_init_with_default_settings(self):
wait client.complete_with_tools("test")
        client = LLMClient()
wait client.complete_with_tools("test")
        assert client.settings is not None
wait client.complete_with_tools("test")
        assert client._anthropic is None
wait client.complete_with_tools("test")
        assert client._openai is None
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
    def test_init_with_custom_settings(self):
wait client.complete_with_tools("test")
        settings = Settings(provider=Provider.OPENAI, model="gpt-4")
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")
        assert client.settings.provider == Provider.OPENAI
wait client.complete_with_tools("test")
        assert client.settings.model == "gpt-4"
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
class TestLLMClientCompleteAnthropic:
wait client.complete_with_tools("test")
    @pytest.mark.asyncio
wait client.complete_with_tools("test")
    async def test_complete_anthropic_mocked(self):
wait client.complete_with_tools("test")
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        mock_response = MagicMock()
wait client.complete_with_tools("test")
        mock_response.content = [MagicMock(type="text", text="Hello, world!")]
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        with patch.object(client, "_anthropic") as mock_anthropic:
wait client.complete_with_tools("test")
            mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
wait client.complete_with_tools("test")
            result = await client.complete("test prompt", system="test system", model="test-model", max_tokens=100)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        assert result == "Hello, world!"
wait client.complete_with_tools("test")
        mock_anthropic.messages.create.assert_called_once()
wait client.complete_with_tools("test")
        call_kwargs = mock_anthropic.messages.create.call_args[1]
wait client.complete_with_tools("test")
        assert call_kwargs["model"] == "test-model"
wait client.complete_with_tools("test")
        assert call_kwargs["max_tokens"] == 100
wait client.complete_with_tools("test")
        assert call_kwargs["system"] == "test system"
wait client.complete_with_tools("test")
        assert call_kwargs["messages"] == [{"role": "user", "content": "test prompt"}]
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
    @pytest.mark.asyncio
wait client.complete_with_tools("test")
    async def test_complete_anthropic_uses_defaults(self):
wait client.complete_with_tools("test")
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key", max_tokens=2048, timeout=60.0)
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        mock_response = MagicMock()
wait client.complete_with_tools("test")
        mock_response.content = [MagicMock(type="text", text="Default response")]
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        with patch.object(client, "_anthropic") as mock_anthropic:
wait client.complete_with_tools("test")
            mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
wait client.complete_with_tools("test")
            await client.complete("test")
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        call_kwargs = mock_anthropic.messages.create.call_args[1]
wait client.complete_with_tools("test")
        assert call_kwargs["max_tokens"] == 2048
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
class TestLLMClientCompleteOpenAI:
wait client.complete_with_tools("test")
    @pytest.mark.asyncio
wait client.complete_with_tools("test")
    async def test_complete_openai_mocked(self):
wait client.complete_with_tools("test")
        settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        mock_message = MagicMock(content="OpenAI response")
wait client.complete_with_tools("test")
        mock_choice = MagicMock(message=mock_message)
wait client.complete_with_tools("test")
        mock_response = MagicMock(choices=[mock_choice])
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        with patch.object(client, "_openai") as mock_openai:
wait client.complete_with_tools("test")
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
wait client.complete_with_tools("test")
            result = await client.complete("test prompt", model="gpt-4", max_tokens=500)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        assert result == "OpenAI response"
wait client.complete_with_tools("test")
        mock_openai.chat.completions.create.assert_called_once()
wait client.complete_with_tools("test")
        call_kwargs = mock_openai.chat.completions.create.call_args[1]
wait client.complete_with_tools("test")
        assert call_kwargs["model"] == "gpt-4"
wait client.complete_with_tools("test")
        assert call_kwargs["max_tokens"] == 500
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
class TestLLMClientFileOperations:
wait client.complete_with_tools("test")
    @pytest.mark.asyncio
wait client.complete_with_tools("test")
    async def test_read_file(self, tmp_path):
wait client.complete_with_tools("test")
        settings = Settings()
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        test_file = tmp_path / "test.txt"
wait client.complete_with_tools("test")
        test_file.write_text("file content")
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        content = await client.read_file(test_file)
wait client.complete_with_tools("test")
        assert content == "file content"
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
    @pytest.mark.asyncio
wait client.complete_with_tools("test")
    async def test_write_file(self, tmp_path):
wait client.complete_with_tools("test")
        settings = Settings()
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        test_file = tmp_path / "new_file.txt"
wait client.complete_with_tools("test")
        await client.write_file(test_file, "new content")
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        assert test_file.read_text() == "new content"
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
    @pytest.mark.asyncio
wait client.complete_with_tools("test")
    async def test_write_file_creates_parent_dirs(self, tmp_path):
wait client.complete_with_tools("test")
        settings = Settings()
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        test_file = tmp_path / "subdir" / "nested" / "file.txt"
wait client.complete_with_tools("test")
        await client.write_file(test_file, "nested content")
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        assert test_file.read_text() == "nested content"
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
    def test_list_files(self, tmp_path):
wait client.complete_with_tools("test")
        settings = Settings()
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        (tmp_path / "file1.txt").write_text("1")
wait client.complete_with_tools("test")
        (tmp_path / "file2.py").write_text("2")
wait client.complete_with_tools("test")
        (tmp_path / "subdir").mkdir()
wait client.complete_with_tools("test")
        (tmp_path / "subdir" / "file3.txt").write_text("3")
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        files = client.list_files(tmp_path)
wait client.complete_with_tools("test")
        assert set(files) == {"file1.txt", "file2.py", "subdir/file3.txt"}
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
    def test_list_files_empty_directory(self, tmp_path):
wait client.complete_with_tools("test")
        settings = Settings()
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        files = client.list_files(tmp_path)
wait client.complete_with_tools("test")
        assert files == []
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
    def test_list_files_nonexistent_directory(self):
wait client.complete_with_tools("test")
        settings = Settings()
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        files = client.list_files("/nonexistent/path")
wait client.complete_with_tools("test")
        assert files == []
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
class TestLLMClientCompleteWithTools:
wait client.complete_with_tools("test")
    @pytest.mark.asyncio
wait client.complete_with_tools("test")
    async def test_complete_with_tools_anthropic(self):
wait client.complete_with_tools("test")
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        mock_tool = MagicMock(type="tool_call", id="call_1")
wait client.complete_with_tools("test")
        mock_tool.tool = MagicMock()
wait client.complete_with_tools("test")
        mock_tool.tool.name = "read_file"
wait client.complete_with_tools("test")
        mock_tool.tool.input = {"path": "test.py"}
wait client.complete_with_tools("test")
        mock_text = MagicMock(type="text", text="I'll read that file")
wait client.complete_with_tools("test")
        mock_response = MagicMock(content=[mock_tool, mock_text])
wait client.complete_with_tools("test")
        mock_response.stop_reason = "tool_use"
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        with patch.object(client, "_anthropic") as mock_anthropic:
wait client.complete_with_tools("test")
            mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
wait client.complete_with_tools("test")
            result = await client.complete_with_tools(
wait client.complete_with_tools("test")
                "test prompt",
wait client.complete_with_tools("test")
                tools=[{"name": "read_file", "description": "Read a file"}],
wait client.complete_with_tools("test")
                tool_choice={"type": "auto"},
wait client.complete_with_tools("test")
            )
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        assert isinstance(result, Completion)
wait client.complete_with_tools("test")
        assert result.text == "I'll read that file"
wait client.complete_with_tools("test")
        assert len(result.tool_calls) == 1
wait client.complete_with_tools("test")
        assert result.tool_calls[0].name == "read_file"
wait client.complete_with_tools("test")
        assert result.tool_calls[0].arguments == {"path": "test.py"}
wait client.complete_with_tools("test")
        assert result.tool_calls[0].id == "call_1"
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
    @pytest.mark.asyncio
wait client.complete_with_tools("test")
    async def test_complete_with_tools_openai(self):
wait client.complete_with_tools("test")
        settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        mock_tool_call = MagicMock()
wait client.complete_with_tools("test")
        mock_tool_call.function = MagicMock()
wait client.complete_with_tools("test")
        mock_tool_call.function.name = "write_file"
wait client.complete_with_tools("test")
        mock_tool_call.function.arguments = '{"path": "test.py", "content": "hi"}'
wait client.complete_with_tools("test")
        mock_tool_call.id = "call_1"
wait client.complete_with_tools("test")
        mock_message = MagicMock(content="Done", tool_calls=[mock_tool_call])
wait client.complete_with_tools("test")
        mock_choice = MagicMock(message=mock_message, finish_reason="tool_calls")
wait client.complete_with_tools("test")
        mock_response = MagicMock(choices=[mock_choice])
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        with patch.object(client, "_openai") as mock_openai:
wait client.complete_with_tools("test")
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
wait client.complete_with_tools("test")
            result = await client.complete_with_tools("test prompt", tools=[{"name": "write_file"}])
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
        assert isinstance(result, Completion)
wait client.complete_with_tools("test")
        assert result.text == "Done"
wait client.complete_with_tools("test")
        assert len(result.tool_calls) == 1
wait client.complete_with_tools("test")
        assert result.tool_calls[0].name == "write_file"
wait client.complete_with_tools("test")
        assert result.tool_calls[0].arguments == {"path": "test.py", "content": "hi"}
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
    @pytest.mark.asyncio
wait client.complete_with_tools("test")
    async def test_complete_with_tools_ollama_fallback(self):
wait client.complete_with_tools("test")
        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")


wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
class TestLLMClientOllamaImportError:
wait client.complete_with_tools("test")
    @pytest.mark.asyncio
wait client.complete_with_tools("test")
    async def test_ollama_raises_import_error_when_not_installed(self):
wait client.complete_with_tools("test")
        settings = Settings(provider=Provider.OLLAMA)
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")
        with patch("furrow.llm.ollama", None), pytest.raises(ImportError, match="ollama package is not installed"):\n            _ = client.ollama_client
wait client.complete_with_tools("test")
                _ = client.ollama_client
wait client.complete_with_tools("test")

wait client.complete_with_tools("test")
    @pytest.mark.asyncio
wait client.complete_with_tools("test")
    async def test_complete_ollama_raises_import_error(self):
wait client.complete_with_tools("test")
        settings = Settings(provider=Provider.OLLAMA)
wait client.complete_with_tools("test")
        client = LLMClient(settings=settings)
wait client.complete_with_tools("test")
        with patch("furrow.llm.ollama", None), pytest.raises(ImportError, match="ollama package is required for OLLAMA provider"):\n            await client.complete("test")
wait client.complete_with_tools("test")
                await client.complete("test")
wait client.complete_with_tools("test")
