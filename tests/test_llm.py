from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def make_settings(provider: Provider = Provider.ANTHROPIC, workspace: Path | None = None) -> Settings:
    kwargs: dict = {}
    if provider == Provider.ANTHROPIC:
        kwargs["anthropic_api_key"] = "test-key"
    elif provider == Provider.OPENAI:
        kwargs["openai_api_key"] = "test-key"
    if workspace is not None:
        kwargs["workspace"] = workspace
    return Settings(provider=provider, **kwargs)


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_command():
    settings = make_settings()
    client = LLMClient(settings=settings)

    # Build a fake process that mimics asyncio.subprocess.Process
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(b"hello stdout", b"hello stderr"))
    proc.returncode = 0

    with patch("asyncio.create_subprocess_shell", AsyncMock(return_value=proc)):
        stdout, stderr, returncode = await client.run_command("echo hi")

    assert stdout == "hello stdout"
    assert stderr == "hello stderr"
    assert returncode == 0


# ---------------------------------------------------------------------------
# read_file / write_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file(tmp_path: Path):
    f = tmp_path / "data.txt"
    f.write_text("hello world")

    client = LLMClient(settings=make_settings(workspace=tmp_path))
    content = await client.read_file(f)
    assert content == "hello world"


@pytest.mark.asyncio
async def test_write_file(tmp_path: Path):
    f = tmp_path / "out.txt"
    client = LLMClient(settings=make_settings(workspace=tmp_path))
    await client.write_file(f, "some content")

    assert f.exists()
    assert f.read_text() == "some content"


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


def test_list_files(tmp_path: Path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("")

    client = LLMClient(settings=make_settings(workspace=tmp_path))
    files = client.list_files(tmp_path)

    assert "a.py" in files
    assert "b.txt" in files
    assert "sub/c.py" in files


def test_list_files_exclude(tmp_path: Path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.pyc").write_text("")

    client = LLMClient(settings=make_settings(workspace=tmp_path))
    files = client.list_files(tmp_path)

    assert "a.py" in files
    assert not any("__pycache__" in f for f in files)
    assert not any(".pyc" in f for f in files)


def test_list_files_nonexistent(tmp_path: Path):
    client = LLMClient(settings=make_settings(workspace=tmp_path))
    files = client.list_files(tmp_path / "does_not_exist")
    assert files == []


def test_list_files_custom_exclude(tmp_path: Path):
    (tmp_path / "keep.py").write_text("")
    (tmp_path / "ignore.log").write_text("")

    client = LLMClient(settings=make_settings(workspace=tmp_path))
    files = client.list_files(tmp_path, exclude={"*.log"})

    assert "keep.py" in files
    assert not any("ignore.log" in f for f in files)


# ---------------------------------------------------------------------------
# _execute_tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_read_file():
    client = LLMClient(settings=make_settings())
    with patch.object(client, "read_file", AsyncMock(return_value="file content")):
        result = await client._execute_tool("read_file", {"path": "x.py"})
    assert result == "file content"


@pytest.mark.asyncio
async def test_execute_tool_write_file():
    client = LLMClient(settings=make_settings())
    with patch.object(client, "write_file", AsyncMock()):
        result = await client._execute_tool("write_file", {"path": "x.py", "content": "hi"})
    assert result == "Successfully wrote to x.py"


@pytest.mark.asyncio
async def test_execute_tool_unknown():
    client = LLMClient(settings=make_settings())
    result = await client._execute_tool("unknown", {})
    assert result == "Unknown tool: unknown"


# ---------------------------------------------------------------------------
# _openai_tools
# ---------------------------------------------------------------------------


def test_openai_tools_format():
    client = LLMClient(settings=make_settings())
    tools = client._openai_tools

    assert isinstance(tools, list)
    assert len(tools) >= 1
    for t in tools:
        assert t["type"] == "function"
        assert "name" in t["function"]
        assert "description" in t["function"]
        assert "parameters" in t["function"]


# ---------------------------------------------------------------------------
# Ollama completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_complete():
    settings = make_settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "ollama said hello"}
    mock_response.raise_for_status = MagicMock()

    mock_httpx = MagicMock()
    mock_httpx.post = AsyncMock(return_value=mock_response)

    # httpx_client is a read-only property; set the backing attribute directly
    client._httpx = mock_httpx
    result = await client._complete_ollama("hi", "sys", "llama3")

    assert result == "ollama said hello"
    mock_httpx.post.assert_awaited_once()


# ---------------------------------------------------------------------------
# Existing tests retained
# ---------------------------------------------------------------------------


def test_read_file_method_exists():
    client = LLMClient(settings=make_settings())
    assert callable(client.read_file)


def test_write_file_method_exists():
    client = LLMClient(settings=make_settings())
    assert callable(client.write_file)