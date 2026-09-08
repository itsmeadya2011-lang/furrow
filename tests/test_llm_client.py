"""Tests for the Furrow LLM client."""

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def tmp_workspace(tmp_path):
    settings = Settings(workspace=tmp_path)
    return settings, tmp_path


class TestLLMClientInit:
    def test_init_with_default_settings(self):
        client = LLMClient()
        assert client.settings is not None

    def test_init_with_custom_settings(self):
        settings = Settings()
        client = LLMClient(settings=settings)
        assert client.settings is settings


class TestLLMClientFileIO:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_workspace):
        settings, tmp_path = tmp_workspace
        client = LLMClient(settings=settings)
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        content = await client.read_file(test_file)
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_workspace):
        settings, tmp_path = tmp_workspace
        client = LLMClient(settings=settings)
        test_file = tmp_path / "subdir" / "test.txt"

        await client.write_file(test_file, "file content")
        assert test_file.exists()
        assert test_file.read_text() == "file content"

    @pytest.mark.asyncio
    async def test_write_file_append_mode(self, tmp_workspace):
        settings, tmp_path = tmp_workspace
        client = LLMClient(settings=settings)
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\n")

        await client.write_file(test_file, "line2\n", mode="a")
        assert test_file.read_text() == "line1\nline2\n"

    @pytest.mark.asyncio
    async def test_list_files(self, tmp_workspace):
        settings, tmp_path = tmp_workspace
        client = LLMClient(settings=settings)
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file2.py").write_text("b")

        files = client.list_files(tmp_path)
        assert "file1.txt" in files
        assert "sub/file2.py" in files

    @pytest.mark.asyncio
    async def test_list_files_nonexistent(self, tmp_workspace):
        settings, tmp_path = tmp_workspace
        client = LLMClient(settings=settings)
        files = client.list_files(tmp_path / "nonexistent")
        assert files == []

    @pytest.mark.asyncio
    async def test_execute_command(self, tmp_workspace):
        settings, _ = tmp_workspace
        client = LLMClient(settings=settings)
        result = await client.execute_command("echo hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_execute_command_timeout(self, tmp_workspace):
        settings, _ = tmp_workspace
        client = LLMClient(settings=settings)
        result = await client.execute_command("sleep 10", timeout=1)
        assert "timed out" in result.lower()


class TestLLMClientOllama:
    @pytest.mark.asyncio
    async def test_complete_ollama(self, tmp_workspace):
        settings, _ = tmp_workspace
        settings.provider = Provider.OLLAMA
        settings.model = "llama3.1"
        client = LLMClient(settings=settings)

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"response": "ollama response"})

        with patch.object(LLMClient, "_ollama_request", new_callable=AsyncMock, return_value=mock_response) as mock_req:
            result = await client._complete_ollama("hello", "", "llama3.1")
            assert result == "ollama response"
            mock_req.assert_called_once()


class TestLLMClientToolCalling:
    def test_tools_defined(self):
        assert len(LLMClient.TOOLS) == 4
        tool_names = [t["function"]["name"] for t in LLMClient.TOOLS]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "list_files" in tool_names
        assert "execute_command" in tool_names

    @pytest.mark.asyncio
    async def test_execute_tool_read_file(self, tmp_workspace):
        settings, tmp_path = tmp_workspace
        client = LLMClient(settings=settings)
        test_file = tmp_path / "tool_test.txt"
        test_file.write_text("tool content")

        result = await client._execute_tool("read_file", {"path": str(test_file)})
        assert result == "tool content"

    @pytest.mark.asyncio
    async def test_execute_tool_write_file(self, tmp_workspace):
        settings, tmp_path = tmp_workspace
        client = LLMClient(settings=settings)
        test_file = tmp_path / "write_test.txt"

        result = await client._execute_tool("write_file", {"path": str(test_file), "content": "written"})
        assert test_file.exists()
        assert "written" in result

    @pytest.mark.asyncio
    async def test_execute_tool_list_files(self, tmp_workspace):
        settings, tmp_path = tmp_workspace
        client = LLMClient(settings=settings)

        result = await client._execute_tool("list_files", {"directory": str(tmp_path)})
        assert "(empty)" in result

    @pytest.mark.asyncio
    async def test_execute_tool_execute_command(self, tmp_workspace):
        settings, _ = tmp_workspace
        client = LLMClient(settings=settings)

        result = await client._execute_tool("execute_command", {"command": "echo tool_test"})
        assert "tool_test" in result

    @pytest.mark.asyncio
    async def test_execute_tool_unknown(self, tmp_workspace):
        settings, _ = tmp_workspace
        client = LLMClient(settings=settings)

        result = await client._execute_tool("unknown_tool", {})
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_execute_tool_file_not_found(self, tmp_workspace):
        settings, _ = tmp_workspace
        client = LLMClient(settings=settings)

        result = await client._execute_tool("read_file", {"path": "/nonexistent/file.txt"})
        assert "not found" in result.lower() or "error" in result.lower()


class TestLLMClientComplete:
    @pytest.mark.asyncio
    async def test_complete_unsupported_provider(self):
        """Calling complete with an unknown provider should raise ValueError."""
        settings = MagicMock(spec=Settings)
        settings.provider = "invalid"
        settings.model = "test"
        client = LLMClient(settings=settings)

        with pytest.raises(ValueError, match="Unsupported provider"):
            await client.complete("test prompt")
