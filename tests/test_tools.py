import pytest

from furrow.agents.tools import (
    MAX_OUTPUT,
    WORKER_TOOLS,
    execute_tool,
)


@pytest.mark.asyncio
async def test_write_file_read_file_roundtrip(tmp_path):
    file_path = tmp_path / "test.txt"
    content = "hello furrow"

    result = await execute_tool("write_file", {"path": str(file_path), "content": content})
    assert result.startswith("OK: wrote")

    result = await execute_tool("read_file", {"path": str(file_path)})
    assert result == content


@pytest.mark.asyncio
async def test_write_file_creates_nested_dirs(tmp_path):
    file_path = tmp_path / "nested" / "dir" / "file.txt"
    content = "nested content"

    result = await execute_tool("write_file", {"path": str(file_path), "content": content})
    assert result.startswith("OK: wrote")

    result = await execute_tool("read_file", {"path": str(file_path)})
    assert result == content


@pytest.mark.asyncio
async def test_list_files_sorted_relative_paths(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "c.txt").write_text("c")

    result = await execute_tool("list_files", {"directory": str(tmp_path)})
    paths = result.split("\n")
    assert paths == ["a.txt", "b.txt", "sub/c.txt"]


@pytest.mark.asyncio
async def test_list_files_empty_dir(tmp_path):
    result = await execute_tool("list_files", {"directory": str(tmp_path)})
    assert result == "(empty)"


@pytest.mark.asyncio
async def test_list_files_missing_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    result = await execute_tool("list_files", {"directory": str(missing)})
    assert result == "ERROR: directory not found"


@pytest.mark.asyncio
async def test_run_command_success():
    result = await execute_tool("run_command", {"command": "echo furrow"})
    assert result.startswith("EXIT 0")
    assert "furrow" in result


@pytest.mark.asyncio
async def test_run_command_timeout():
    result = await execute_tool("run_command", {"command": "sleep 5", "timeout": 1})
    assert result == "ERROR: command timed out after 1s"


@pytest.mark.asyncio
async def test_unknown_tool():
    result = await execute_tool("unknown_tool", {})
    assert result == "ERROR: unknown tool: unknown_tool"


@pytest.mark.asyncio
async def test_read_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.txt"
    result = await execute_tool("read_file", {"path": str(missing)})
    assert result.startswith("ERROR:")


def test_worker_tools_structure():
    assert len(WORKER_TOOLS) == 4
    tool_names = {t["name"] for t in WORKER_TOOLS}
    assert tool_names == {"read_file", "write_file", "list_files", "run_command"}

    for tool in WORKER_TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        assert isinstance(tool["input_schema"], dict)
        assert tool["input_schema"]["type"] == "object"
        assert "properties" in tool["input_schema"]
        assert "required" in tool["input_schema"]


def test_max_output_constant():
    assert MAX_OUTPUT == 10_000