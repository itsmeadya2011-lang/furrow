"""Tests for LLMClient tool definitions and execution."""
import json
import os
import tempfile
from pathlib import Path

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient, _anthropic_tools, _openai_tools


@pytest.fixture
def client():
    return LLMClient(settings=Settings(provider=Provider.OLLAMA))


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def test_anthropic_tools_format():
    tools = _anthropic_tools()
    names = [t["name"] for t in tools]
    assert "read_file" in names
    assert "write_file" in names
    assert "list_files" in names
    # Anthropic uses input_schema
    for t in tools:
        assert "input_schema" in t
        assert "name" in t
        assert "description" in t


def test_openai_tools_format():
    tools = _openai_tools()
    names = [t["function"]["name"] for t in tools]
    assert "read_file" in names
    assert "write_file" in names
    assert "list_files" in names
    # OpenAI uses parameters + function wrapper
    for t in tools:
        assert t["type"] == "function"
        assert "parameters" in t["function"]
        assert "description" in t["function"]


def test_tool_schemas_have_required_fields():
    from furrow.llm import _TOOL_SCHEMAS

    for name, schema in _TOOL_SCHEMAS.items():
        assert "type" in schema and schema["type"] == "object"
        assert "properties" in schema
        assert isinstance(schema.get("required"), list) and len(schema["required"]) > 0


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_tool_write_then_read(client, tmp_path):
    f = tmp_path / "hello.txt"
    result = await client._execute_tool("write_file", {"path": str(f), "content": "hi there"})
    assert "Written" in result
    assert f.read_text() == "hi there"

    result = await client._execute_tool("read_file", {"path": str(f)})
    assert result == "hi there"


@pytest.mark.asyncio
async def test_tool_read_file_not_found(client):
    result = await client._tool_read_file("/nonexistent/path/file.txt")
    assert "not found" in result.lower() or "error" in result.lower()


@pytest.mark.asyncio
async def test_tool_list_files(client, tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("y")
    result = await client._tool_list_files(str(tmp_path))
    files = json.loads(result)
    assert "a.py" in files
    assert "b.py" in files


@pytest.mark.asyncio
async def test_tool_list_files_nonexistent(client):
    result = await client._tool_list_files("/nonexistent/dir")
    assert json.loads(result) == []


@pytest.mark.asyncio
async def test_execute_unknown_tool(client):
    result = await client._execute_tool("nonexistent_tool", {"x": 1})
    assert "unknown tool" in result


@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(client, tmp_path):
    f = tmp_path / "nested" / "deep" / "file.py"
    result = await client._tool_write_file(str(f), "content")
    assert f.exists()
    assert f.read_text() == "content"


# ---------------------------------------------------------------------------
# Provider handling
# ---------------------------------------------------------------------------

def test_ollama_client_uses_base_url():
    s = Settings(provider=Provider.OLLAMA)
    c = LLMClient(settings=s)
    # Ollama client should be created with the right base_url
    assert str(c.ollama.base_url) == s.ollama_base_url


def test_openai_client_raises_without_key():
    s = Settings(provider=Provider.OPENAI, openai_api_key=None)
    c = LLMClient(settings=s)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = c.openai


def test_anthropic_client_raises_without_key():
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    c = LLMClient(settings=s)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = c.anthropic


def test_max_tool_calls_setting():
    s = Settings()
    assert s.max_tool_calls > 0
