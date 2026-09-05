"""Tests for the tools module."""
import pytest
from pathlib import Path

from furrow.tools.base import Tool, ToolResult
from furrow.tools.file_ops import ReadFile, WriteFile, EditFile, ListFiles, GrepFiles
from furrow.tools.execute import ExecuteCommand


class TestToolResult:
    def test_default_success(self):
        tr = ToolResult()
        assert tr.success is True
        assert tr.output == ""
        assert tr.error is None

    def test_with_error(self):
        tr = ToolResult(success=False, error="something broke")
        assert tr.success is False
        assert tr.error == "something broke"


class TestToolSchema:
    def test_read_file_schema(self):
        schema = ReadFile.to_anthropic_schema()
        assert schema["name"] == "read_file"
        assert "input_schema" in schema
        assert "path" in schema["input_schema"]["required"]

    def test_write_file_openai_schema(self):
        schema = WriteFile.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "write_file"
        assert "path" in schema["function"]["parameters"]["required"]
        assert "content" in schema["function"]["parameters"]["required"]

    def test_edit_file_schema(self):
        schema = EditFile.to_anthropic_schema()
        assert "input_schema" in schema
        props = schema["input_schema"]["properties"]
        assert "path" in props
        assert "old_str" in props
        assert "new_str" in props
        assert set(schema["input_schema"]["required"]) == {"path", "old_str", "new_str"}


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await ReadFile.execute(path=str(f))
        assert isinstance(result, ToolResult)
        assert result.success
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        result = await ReadFile.execute(path="/nonexistent/file.txt")
        assert isinstance(result, ToolResult)
        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_read_with_line_range(self, tmp_path):
        lines = [f"line {i}\n" for i in range(10)]
        f = tmp_path / "lines.txt"
        f.write_text("".join(lines))
        result = await ReadFile.execute(path=str(f), line_start=2, line_end=5)
        assert result.success
        assert "line 2" in result.output
        assert "line 4" in result.output
        assert "line 5" not in result.output


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path):
        f = tmp_path / "newdir" / "test.txt"
        result = await WriteFile.execute(path=str(f), content="test content")
        assert result.success
        assert f.read_text() == "test content"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "a" / "b" / "c" / "test.txt"
        result = await WriteFile.execute(path=str(f), content="nested")
        assert result.success
        assert f.exists()

    @pytest.mark.asyncio
    async def test_write_overwrite(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("original")
        await WriteFile.execute(path=str(f), content="overwritten")
        assert f.read_text() == "overwritten"


class TestEditFile:
    @pytest.mark.asyncio
    async def test_edit_existing_string(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    pass\n")
        result = await EditFile.execute(
            path=str(f),
            old_str="def hello():",
            new_str="def hello(name):",
        )
        assert result.success
        content = f.read_text()
        assert "def hello(name):" in content

    @pytest.mark.asyncio
    async def test_edit_nonexistent_file(self):
        result = await EditFile.execute(
            path="/nonexistent/file.txt",
            old_str="old",
            new_str="new",
        )
        assert not result.success

    @pytest.mark.asyncio
    async def test_edit_string_not_found(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await EditFile.execute(
            path=str(f),
            old_str="nonexistent",
            new_str="replacement",
        )
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_edit_multiple_matches(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("foo\nfoo\nfoo")
        result = await EditFile.execute(
            path=str(f),
            old_str="foo",
            new_str="bar",
        )
        assert not result.success
        assert "3 times" in result.error


class TestListFiles:
    @pytest.mark.asyncio
    async def test_list_files_recursive(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("b")
        result = await ListFiles.execute(path=str(tmp_path), recursive=True)
        assert result.success
        assert "a.txt" in result.output

    @pytest.mark.asyncio
    async def test_list_files_nonexistent(self):
        result = await ListFiles.execute(path="/nonexistent/dir")
        assert not result.success


class TestGrepFiles:
    @pytest.mark.asyncio
    async def test_grep_finds_pattern(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")
        result = await GrepFiles.execute(pattern="def ", path=str(tmp_path))
        assert result.success
        assert "test.py" in result.output

    @pytest.mark.asyncio
    async def test_grep_no_matches(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("hello world")
        result = await GrepFiles.execute(pattern="zzz", path=str(tmp_path))
        assert "no matches" in result.output.lower()


class TestExecuteCommand:
    @pytest.mark.asyncio
    async def test_execute_simple_command(self):
        result = await ExecuteCommand.execute(command="echo hello")
        assert result.success
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_execute_command_not_found(self):
        result = await ExecuteCommand.execute(command="nonexistent_command_12345")
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_command_timeout(self):
        result = await ExecuteCommand.execute(command="sleep 10", timeout=1)
        assert not result.success
        assert "timed out" in result.error.lower()


class TestDefaultTools:
    def test_default_tools_list(self):
        from furrow.tools import DEFAULT_TOOLS
        names = [t.name for t in DEFAULT_TOOLS]
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "list_files" in names
        assert "grep_files" in names
        assert "execute_command" in names

    def test_all_tools_have_name_and_description(self):
        from furrow.tools import DEFAULT_TOOLS
        for tool in DEFAULT_TOOLS:
            assert tool.name, f"Tool missing name: {tool}"
            assert tool.description, f"Tool missing description: {tool}"

    def test_annotation_to_schema(self):
        from furrow.tools.base import _annotation_to_schema
        import typing

        assert _annotation_to_schema(str) == "string"
        assert _annotation_to_schema(int) == "integer"
        assert _annotation_to_schema(float) == "number"
        assert _annotation_to_schema(bool) == "boolean"
        assert _annotation_to_schema(list) == "array"
        assert _annotation_to_schema(dict) == "object"
        # Optional types
        assert _annotation_to_schema(typing.Optional[int]) == "integer"
        # Python 3.10+ union syntax
        assert _annotation_to_schema(int | None) == "integer"

    def test_tool_schema_consistency(self):
        """Anthropic and OpenAI schemas should have same params."""
        for tool_cls in [ReadFile, WriteFile, EditFile, ListFiles, GrepFiles, ExecuteCommand]:
            anthropic = tool_cls.to_anthropic_schema()
            openai = tool_cls.to_openai_schema()
            assert anthropic["name"] == openai["function"]["name"]
            assert anthropic["description"] == openai["function"]["description"]
            assert anthropic["input_schema"] == openai["function"]["parameters"]

