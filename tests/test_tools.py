"""Tests for the Furrow tools module."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from furrow.tools import GrepTool, ListFilesTool, ReadFileTool, ShellTool, WriteFileTool, get_default_tools
from furrow.tools.base import Tool, ToolResult


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    return tmp_path


@pytest.fixture
def sample_file(tmp_workspace: Path) -> Path:
    """Create a sample file in the temp workspace."""
    file_path = tmp_workspace / "test.py"
    file_path.write_text("def hello():\n    return 'world'\n")
    return file_path


# --- ToolResult ---


def test_tool_result_success():
    result = ToolResult(success=True, output="ok")
    assert result.success is True
    assert result.output == "ok"
    assert result.error is None


def test_tool_result_failure():
    result = ToolResult(success=False, output="", error="something broke")
    assert result.success is False
    assert result.error == "something broke"


# --- Tool base class ---


def test_tool_is_abstract():
    """Tool should not be directly instantiable."""
    with pytest.raises(TypeError):
        Tool()  # type: ignore[abstract]


def test_tool_schema_methods():
    """Tool subclasses should provide anthropic and openai schemas."""

    class TestTool(Tool):
        @property
        def name(self) -> str:
            return "test_tool"

        @property
        def description(self) -> str:
            return "A test tool"

        @property
        def _properties(self) -> dict:
            return {"foo": {"type": "string"}}

        @property
        def _required(self) -> list[str]:
            return ["foo"]

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="done")

    t = TestTool()
    anthropic_schema = t.to_anthropic_schema()
    assert anthropic_schema["name"] == "test_tool"
    assert anthropic_schema["input_schema"]["properties"]["foo"]["type"] == "string"
    assert "foo" in anthropic_schema["input_schema"]["required"]

    openai_schema = t.to_openai_schema()
    assert openai_schema["type"] == "function"
    assert openai_schema["function"]["name"] == "test_tool"
    assert "foo" in openai_schema["function"]["parameters"]["properties"]


# --- ReadFileTool ---


@pytest.mark.asyncio
async def test_read_file_success(sample_file: Path):
    tool = ReadFileTool()
    result = await tool.execute(path=str(sample_file))
    assert result.success is True
    assert "def hello" in result.output


@pytest.mark.asyncio
async def test_read_file_not_found(tmp_workspace: Path):
    tool = ReadFileTool()
    result = await tool.execute(path="nonexistent.py")
    assert result.success is False
    assert "File not found" in (result.error or "")


@pytest.mark.asyncio
async def test_read_file_missing_path():
    tool = ReadFileTool()
    result = await tool.execute()
    assert result.success is False
    assert "Missing" in (result.error or "")


# --- WriteFileTool ---


@pytest.mark.asyncio
async def test_write_file_creates_file(tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_workspace)
    tool = WriteFileTool()
    result = await tool.execute(path="output.txt", content="hello world")
    assert result.success is True
    written = (tmp_workspace / "output.txt").read_text()
    assert written == "hello world"


@pytest.mark.asyncio
async def test_write_file_creates_directories(tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_workspace)
    tool = WriteFileTool()
    result = await tool.execute(path="nested/deep/file.txt", content="content")
    assert result.success is True
    assert (tmp_workspace / "nested/deep/file.txt").read_text() == "content"


@pytest.mark.asyncio
async def test_write_file_missing_args():
    tool = WriteFileTool()
    result = await tool.execute(path="test.txt")
    assert result.success is False


# --- ListFilesTool ---


@pytest.mark.asyncio
async def test_list_files(tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_workspace)
    (tmp_workspace / "a.py").write_text("a")
    (tmp_workspace / "b.py").write_text("b")
    (tmp_workspace / "sub").mkdir()
    (tmp_workspace / "sub" / "c.py").write_text("c")

    tool = ListFilesTool()
    result = await tool.execute(directory=".")
    assert result.success is True
    assert "a.py" in result.output
    assert "b.py" in result.output
    assert "sub/c.py" in result.output


@pytest.mark.asyncio
async def test_list_files_default_directory(tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_workspace)
    (tmp_workspace / "x.py").write_text("x")

    tool = ListFilesTool()
    result = await tool.execute()
    assert result.success is True
    assert "x.py" in result.output


@pytest.mark.asyncio
async def test_list_files_not_found():
    tool = ListFilesTool()
    result = await tool.execute(directory="/nonexistent/path")
    assert result.success is False


# --- GrepTool ---


@pytest.mark.asyncio
async def test_grep_tool(tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_workspace)
    (tmp_workspace / "main.py").write_text("def foo():\n    return 'bar'\n\ndef bar():\n    return 'foo'\n")
    (tmp_workspace / "other.py").write_text("print('foo')\n")

    tool = GrepTool()
    result = await tool.execute(pattern="def foo", directory=".")
    assert result.success is True
    assert "main.py" in result.output


@pytest.mark.asyncio
async def test_grep_no_matches(tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_workspace)
    (tmp_workspace / "main.py").write_text("hello")

    tool = GrepTool()
    result = await tool.execute(pattern="nonexistent_pattern", directory=".")
    assert result.success is True
    assert "No matches" in result.output


@pytest.mark.asyncio
async def test_grep_invalid_regex(tmp_workspace: Path):
    tool = GrepTool()
    result = await tool.execute(pattern="[invalid")
    assert result.success is False


# --- ShellTool ---


@pytest.mark.asyncio
async def test_shell_success():
    tool = ShellTool()
    result = await tool.execute(command="echo hello")
    assert result.success is True
    assert "hello" in result.output
    assert "[exit code: 0]" in result.output


@pytest.mark.asyncio
async def test_shell_failure():
    tool = ShellTool()
    result = await tool.execute(command="exit 1")
    assert result.success is False


@pytest.mark.asyncio
async def test_shell_missing_command():
    tool = ShellTool()
    result = await tool.execute()
    assert result.success is False
    assert "Missing" in (result.error or "")


@pytest.mark.asyncio
async def test_shell_timeout():
    tool = ShellTool()
    result = await tool.execute(command="sleep 10", timeout=1)
    assert result.success is False
    assert "timed out" in (result.error or "")


# --- get_default_tools ---


def test_get_default_tools():
    tools = get_default_tools()
    assert len(tools) == 5
    names = {t.name for t in tools}
    assert names == {"read_file", "write_file", "list_files", "grep", "shell"}


def test_default_tools_are_tool_instances():
    tools = get_default_tools()
    for t in tools:
        assert isinstance(t, Tool)
