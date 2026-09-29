"""Tests for the LLMClient, focusing on tool execution and file I/O helpers."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace with sample files."""
    (tmp_path / "existing.txt").write_text("hello world")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.py").write_text("print('hi')")
    return tmp_path


@pytest.fixture
def client(temp_workspace):
    """Create an LLMClient with a temp workspace."""
    s = Settings(workspace=temp_workspace, provider=Provider.OPENAI, openai_api_key="test-key")
    c = LLMClient(settings=s)
    return c, temp_workspace


class TestFileIO:
    @pytest.mark.asyncio
    async def test_read_file(self, client):
        c, ws = client
        content = await c.read_file(ws / "existing.txt")
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_write_file_creates_parents(self, client):
        c, ws = client
        await c.write_file(ws / "new_dir" / "deep" / "file.txt", "content here")
        assert (ws / "new_dir" / "deep" / "file.txt").read_text() == "content here"

    @pytest.mark.asyncio
    async def test_list_files(self, client):
        c, ws = client
        files = c.list_files(ws)
        assert "existing.txt" in files
        assert "sub/nested.py" in files

    @pytest.mark.asyncio
    async def test_list_files_nonexistent(self, client):
        c, ws = client
        assert c.list_files(ws / "nonexistent") == []


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_execute_read_file_tool(self, client):
        c, ws = client
        result = await c._execute_tool("read_file", {"path": str(ws / "existing.txt")})
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_execute_write_file_tool(self, client):
        c, ws = client
        result = await c._execute_tool("write_file", {"path": str(ws / "new.txt"), "content": "data"})
        assert "File written" in result
        assert (ws / "new.txt").read_text() == "data"

    @pytest.mark.asyncio
    async def test_execute_list_files_tool(self, client):
        c, ws = client
        result = await c._execute_tool("list_files", {"directory": str(ws)})
        assert "existing.txt" in result

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, client):
        c, ws = client  # noqa: RUF059
        result = await c._execute_tool("nonexistent_tool", {})
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_execute_tool_handles_error(self, client):
        c, ws = client  # noqa: RUF059
        result = await c._execute_tool("read_file", {"path": "/nonexistent/path/file.txt"})
        assert "Tool error" in result


class TestOllamaClient:
    @pytest.mark.asyncio
    async def test_ollama_complete(self, tmp_path):
        s = Settings(workspace=tmp_path, provider=Provider.OLLAMA)
        c = LLMClient(settings=s)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ollama response"}}
        mock_resp.raise_for_status.return_value = None

        mock_httpx = AsyncMock()
        mock_httpx.post = AsyncMock(return_value=mock_resp)
        c._httpx = mock_httpx

        result = await c.complete("hello", system="", model="llama3")
        assert result == "ollama response"

    @pytest.mark.asyncio
    async def test_openai_complete(self, tmp_path):
        s = Settings(workspace=tmp_path, provider=Provider.OPENAI, openai_api_key="sk-test")
        c = LLMClient(settings=s)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="openai response"))]

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        c._openai = mock_openai

        result = await c.complete("hello", system="sys", model="gpt-4")
        assert result == "openai response"


class TestToolSchema:
    def test_worker_tool_schema_has_three_tools(self):
        from furrow.agents.worker import FILE_TOOLS

        assert len(FILE_TOOLS) == 3
        names = {t["name"] for t in FILE_TOOLS}
        assert names == {"read_file", "write_file", "list_files"}

    def test_worker_tool_schema_has_required_fields(self):
        from furrow.agents.worker import FILE_TOOLS

        for tool in FILE_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert "properties" in tool["input_schema"]
