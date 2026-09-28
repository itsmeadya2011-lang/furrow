from __future__ import annotations

import json

import pytest

from furrow.llm import ToolCall
from furrow.tools import TOOL_SCHEMAS, ToolRunner


def _runner(tmp_path):
    return ToolRunner(workspace=str(tmp_path))


def test_tool_schemas_have_five_tools():
    names = {s["name"] for s in TOOL_SCHEMAS}
    assert names == {"read_file", "write_file", "edit_file", "list_files", "run_command"}


def test_get_tool_names():
    r = _runner(__file__)
    assert r.get_tool_names() == [
        "read_file",
        "write_file",
        "edit_file",
        "list_files",
        "run_command",
    ]


@pytest.mark.asyncio
async def test_write_and_read_file(tmp_path):
    runner = _runner(tmp_path)
    res = await runner.execute(
        ToolCall(id="1", name="write_file", args={"path": "a/b.txt", "content": "hello"})
    )
    assert res.startswith("Wrote")
    res2 = await runner.execute(ToolCall(id="2", name="read_file", args={"path": "a/b.txt"}))
    assert res2 == "hello"


@pytest.mark.asyncio
async def test_edit_file(tmp_path):
    runner = _runner(tmp_path)
    await runner.execute(
        ToolCall(id="1", name="write_file", args={"path": "f.txt", "content": "foo bar foo"})
    )
    res = await runner.execute(
        ToolCall(
            id="2",
            name="edit_file",
            args={"path": "f.txt", "old_string": "foo", "new_string": "baz"},
        )
    )
    assert "matched 2 occurrence(s)" in res
    text = await runner.execute(ToolCall(id="3", name="read_file", args={"path": "f.txt"}))
    assert text == "baz bar foo"


@pytest.mark.asyncio
async def test_edit_file_missing_old_string(tmp_path):
    runner = _runner(tmp_path)
    await runner.execute(
        ToolCall(id="1", name="write_file", args={"path": "f.txt", "content": "hello"})
    )
    res = await runner.execute(
        ToolCall(
            id="2",
            name="edit_file",
            args={"path": "f.txt", "old_string": "nope", "new_string": "x"},
        )
    )
    assert "Error:" in res


@pytest.mark.asyncio
async def test_list_files(tmp_path):
    runner = _runner(tmp_path)
    await runner.execute(ToolCall(id="1", name="write_file", args={"path": "x.py", "content": "x"}))
    await runner.execute(
        ToolCall(id="2", name="write_file", args={"path": "sub/y.py", "content": "y"})
    )
    res = await runner.execute(ToolCall(id="3", name="list_files", args={"path": "."}))
    files = json.loads(res)
    assert set(files) == {"x.py", "sub/y.py"}


@pytest.mark.asyncio
async def test_list_files_default_path(tmp_path):
    runner = _runner(tmp_path)
    await runner.execute(ToolCall(id="1", name="write_file", args={"path": "a.py", "content": "a"}))
    res = await runner.execute(ToolCall(id="2", name="list_files", args={}))
    assert "a.py" in json.loads(res)


@pytest.mark.asyncio
async def test_run_command(tmp_path):
    runner = _runner(tmp_path)
    res = await runner.execute(ToolCall(id="1", name="run_command", args={"command": "echo hello"}))
    assert "hello" in res


@pytest.mark.asyncio
async def test_run_command_timeout(tmp_path):
    runner = _runner(tmp_path)
    res = await runner.execute(
        ToolCall(id="1", name="run_command", args={"command": "sleep 5", "timeout": 0.5})
    )
    assert "Timeout" in res


@pytest.mark.asyncio
async def test_unknown_tool(tmp_path):
    runner = _runner(tmp_path)
    res = await runner.execute(ToolCall(id="1", name="nope", args={}))
    assert res == "Error: unknown tool 'nope'"


@pytest.mark.asyncio
async def test_resolve_absolute_path(tmp_path):
    runner = _runner(tmp_path)
    p = tmp_path / "abs.txt"
    p.write_text("abs")
    res = await runner.execute(ToolCall(id="1", name="read_file", args={"path": str(p)}))
    assert res == "abs"


@pytest.mark.asyncio
async def test_run_command_cwd_is_workspace(tmp_path):
    runner = _runner(tmp_path)
    res = await runner.execute(ToolCall(id="1", name="run_command", args={"command": "pwd"}))
    assert res.strip() == str(tmp_path.resolve())
