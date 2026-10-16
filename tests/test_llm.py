import pytest

from furrow.llm import FILE_TOOLS, ToolCall, ToolResult


def test_tool_result_defaults():
    result = ToolResult()
    assert result.content == ""
    assert result.tool_calls == []


def test_tool_result_with_values():
    result = ToolResult(content="done", tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "a.txt"})])
    assert result.content == "done"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"


def test_file_tools_expose_read_write_list():
    names = {tool["name"] for tool in FILE_TOOLS}
    assert {"read_file", "write_file", "list_files"} <= names


@pytest.mark.asyncio
async def test_execute_tool_write_then_read(tmp_path):
    from furrow.llm import LLMClient

    client = LLMClient()
    target = tmp_path / "sub" / "hello.txt"
    await client.execute_tool("write_file", {"path": str(target), "content": "hello world"})
    assert target.read_text() == "hello world"

    content = await client.execute_tool("read_file", {"path": str(target)})
    assert content == "hello world"


@pytest.mark.asyncio
async def test_execute_tool_list_files(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    from furrow.llm import LLMClient

    client = LLMClient()
    result = await client.execute_tool("list_files", {"directory": str(tmp_path)})
    assert "a.py" in result
    assert "b.py" in result


@pytest.mark.asyncio
async def test_execute_tool_unknown_raises():
    from furrow.llm import LLMClient

    client = LLMClient()
    with pytest.raises(ValueError, match="Unknown tool"):
        await client.execute_tool("not_a_tool", {})
