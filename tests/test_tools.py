import pytest
import tempfile
from pathlib import Path

from furrow.tools import (
    execute_tool,
    get_tool_schemas,
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ListFilesTool,
    RunCommandTool,
    GlobTool,
    GrepTool,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.mark.asyncio
async def test_read_file_tool(temp_dir):
    """Test reading a file."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("Hello, World!")

    result = await execute_tool("read_file", path=str(test_file))
    assert result.success
    assert result.output == "Hello, World!"


@pytest.mark.asyncio
async def test_read_file_not_found(temp_dir):
    """Test reading a non-existent file."""
    result = await execute_tool("read_file", path=str(temp_dir / "nonexistent.txt"))
    assert not result.success
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_write_file_tool(temp_dir):
    """Test writing a file."""
    test_file = temp_dir / "new.txt"

    result = await execute_tool("write_file", path=str(test_file), content="New content")
    assert result.success
    assert test_file.read_text() == "New content"


@pytest.mark.asyncio
async def test_write_file_creates_dirs(temp_dir):
    """Test writing a file creates parent directories."""
    test_file = temp_dir / "subdir" / "nested" / "file.txt"

    result = await execute_tool("write_file", path=str(test_file), content="Nested")
    assert result.success
    assert test_file.read_text() == "Nested"


@pytest.mark.asyncio
async def test_edit_file_tool(temp_dir):
    """Test editing a file."""
    test_file = temp_dir / "edit.txt"
    test_file.write_text("Original content")

    result = await execute_tool("edit_file", path=str(test_file), old_string="Original", new_string="Modified")
    assert result.success
    assert test_file.read_text() == "Modified content"


@pytest.mark.asyncio
async def test_edit_file_not_found(temp_dir):
    """Test editing non-existent file."""
    result = await execute_tool("edit_file", path=str(temp_dir / "nonexistent.txt"), old_string="a", new_string="b")
    assert not result.success


@pytest.mark.asyncio
async def test_edit_file_string_not_found(temp_dir):
    """Test editing when old string not in file."""
    test_file = temp_dir / "edit.txt"
    test_file.write_text("Hello")

    result = await execute_tool("edit_file", path=str(test_file), old_string="World", new_string="There")
    assert not result.success


@pytest.mark.asyncio
async def test_list_files_tool(temp_dir):
    """Test listing files."""
    (temp_dir / "file1.txt").write_text("1")
    (temp_dir / "file2.txt").write_text("2")
    (temp_dir / "subdir").mkdir()
    (temp_dir / "subdir" / "file3.txt").write_text("3")

    result = await execute_tool("list_files", path=str(temp_dir))
    assert result.success
    assert "file1.txt" in result.output
    assert "file2.txt" in result.output
    assert "subdir/file3.txt" in result.output


@pytest.mark.asyncio
async def test_list_files_with_pattern(temp_dir):
    """Test listing files with pattern."""
    (temp_dir / "test.py").write_text("py")
    (temp_dir / "test.txt").write_text("txt")

    result = await execute_tool("list_files", path=str(temp_dir), pattern="*.py")
    assert result.success
    assert "test.py" in result.output
    assert "test.txt" not in result.output


@pytest.mark.asyncio
async def test_run_command_tool(temp_dir):
    """Test running a command."""
    result = await execute_tool("run_command", command="echo hello", cwd=str(temp_dir))
    assert result.success
    assert "hello" in result.output


@pytest.mark.asyncio
async def test_run_command_failure(temp_dir):
    """Test running a failing command."""
    result = await execute_tool("run_command", command="exit 1", cwd=str(temp_dir))
    assert not result.success


@pytest.mark.asyncio
async def test_glob_tool(temp_dir):
    """Test glob pattern matching."""
    (temp_dir / "test.py").write_text("py")
    (temp_dir / "sub").mkdir()
    (temp_dir / "sub" / "test.py").write_text("py")
    (temp_dir / "test.txt").write_text("txt")

    result = await execute_tool("glob", pattern="**/*.py", path=str(temp_dir))
    assert result.success
    assert "test.py" in result.output
    assert "sub/test.py" in result.output
    assert "test.txt" not in result.output


@pytest.mark.asyncio
async def test_grep_tool(temp_dir):
    """Test grep search."""
    (temp_dir / "file1.py").write_text("def hello():\n    pass")
    (temp_dir / "file2.py").write_text("def world():\n    pass")
    (temp_dir / "file3.txt").write_text("hello world")

    result = await execute_tool("grep", pattern="def ", path=str(temp_dir), include="*.py")
    assert result.success
    assert "file1.py" in result.output
    assert "file2.py" in result.output
    assert "hello()" in result.output
    assert "world()" in result.output


def test_get_tool_schemas():
    """Test getting tool schemas."""
    schemas = get_tool_schemas()
    assert len(schemas) == 7

    names = {s["name"] for s in schemas}
    expected = {"read_file", "write_file", "edit_file", "list_files", "run_command", "glob", "grep"}
    assert names == expected

    for schema in schemas:
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema


if __name__ == "__main__":
    pytest.main([__file__, "-v"])