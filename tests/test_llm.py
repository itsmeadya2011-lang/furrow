from unittest.mock import MagicMock

from furrow.config import Provider
from furrow.llm import LLMClient


def test_format_tools_anthropic():
    client = LLMClient()
    tools = client._format_tools_for_provider(Provider.ANTHROPIC)
    assert len(tools) == len(client.TOOLS)
    first = tools[0]
    assert first["type"] == "function"
    assert first["name"] == "read_file"
    assert first["description"] == client.TOOLS[0]["description"]
    assert first["input_schema"] == client.TOOLS[0]["input_schema"]


def test_format_tools_openai():
    client = LLMClient()
    tools = client._format_tools_for_provider(Provider.OPENAI)
    first = tools[0]
    assert first["type"] == "function"
    assert first["function"]["name"] == "read_file"
    assert first["function"]["parameters"] == client.TOOLS[0]["input_schema"]


def test_format_tools_ollama():
    client = LLMClient()
    tools = client._format_tools_for_provider(Provider.OLLAMA)
    first = tools[0]
    assert first["function"]["name"] == "read_file"
    assert first["function"]["parameters"] == client.TOOLS[0]["input_schema"]


def test_format_tools_unknown_provider_returns_empty():
    client = LLMClient()
    assert client._format_tools_for_provider(MagicMock()) == []


async def test_call_tool_write_file(tmp_path):
    client = LLMClient()
    target = tmp_path / "out.txt"
    result, is_error = await client._call_tool(
        "write_file", path=str(target), content="hello"
    )
    assert is_error is False
    assert target.read_text() == "hello"
    assert "written" in result


async def test_call_tool_read_file(tmp_path):
    path = tmp_path / "in.txt"
    path.write_text("hello")
    client = LLMClient()
    result, is_error = await client._call_tool("read_file", path=str(path))
    assert is_error is False
    assert result == "hello"


async def test_call_tool_unknown_tool():
    client = LLMClient()
    result, is_error = await client._call_tool("does_not_exist")
    assert is_error is True
    assert "Unknown tool" in result
