from __future__ import annotations

from furrow.agents.worker import TOOLS, WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient, ToolCall


def _client(workspace):
    return LLMClient(settings=Settings(workspace=workspace))


def _task():
    return TaskModel(id="1", description="do something", files=["a.py"])


async def test_run_calls_tool_loop_with_three_tools(tmp_path):
    client = _client(tmp_path)
    recorded = {}

    async def fake_run_tool_loop(**kwargs):
        recorded.update(kwargs)
        return "done"

    client.run_tool_loop = fake_run_tool_loop
    worker = WorkerAgent(_task(), client=client)
    result = await worker.run()
    assert result == "done"
    assert recorded["tools"] == TOOLS
    assert [t["name"] for t in recorded["tools"]] == [
        "read_file",
        "write_file",
        "list_files",
    ]
    assert recorded["model"] == client.settings.worker_model
    assert recorded["max_rounds"] == 8
    assert recorded["system"]


async def test_write_file_creates_content(tmp_path):
    client = _client(tmp_path)
    worker = WorkerAgent(_task(), client=client)
    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"path": "sub/out.txt", "content": "hello"},
    )
    msg = await worker._execute_tool(call)
    assert msg == "Wrote sub/out.txt"
    assert (tmp_path / "sub" / "out.txt").read_text() == "hello"


async def test_write_escape_relative_refused(tmp_path):
    client = _client(tmp_path)
    worker = WorkerAgent(_task(), client=client)
    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"path": "../escape.txt", "content": "x"},
    )
    msg = await worker._execute_tool(call)
    assert "Refused" in msg
    assert not (tmp_path.parent / "escape.txt").exists()


async def test_write_absolute_outside_refused(tmp_path):
    client = _client(tmp_path)
    worker = WorkerAgent(_task(), client=client)
    target = tmp_path.parent / ("unique_escape_target_" + str(id(tmp_path)))
    call = ToolCall(
        id="1",
        name="write_file",
        arguments={"path": str(target), "content": "x"},
    )
    msg = await worker._execute_tool(call)
    assert "Refused" in msg
    assert not target.exists()


async def test_read_missing(tmp_path):
    client = _client(tmp_path)
    worker = WorkerAgent(_task(), client=client)
    call = ToolCall(
        id="1",
        name="read_file",
        arguments={"path": "nope.txt"},
    )
    msg = await worker._execute_tool(call)
    assert "File not found" in msg


async def test_read_existing(tmp_path):
    (tmp_path / "a.txt").write_text("contents")
    client = _client(tmp_path)
    worker = WorkerAgent(_task(), client=client)
    call = ToolCall(
        id="1",
        name="read_file",
        arguments={"path": "a.txt"},
    )
    msg = await worker._execute_tool(call)
    assert msg == "contents"


async def test_list_files(tmp_path):
    (tmp_path / "x.py").write_text("y")
    client = _client(tmp_path)
    worker = WorkerAgent(_task(), client=client)
    call = ToolCall(
        id="1",
        name="list_files",
        arguments={},
    )
    msg = await worker._execute_tool(call)
    assert "x.py" in msg


async def test_unknown_tool(tmp_path):
    client = _client(tmp_path)
    worker = WorkerAgent(_task(), client=client)
    call = ToolCall(
        id="1",
        name="frobnicate",
        arguments={},
    )
    msg = await worker._execute_tool(call)
    assert "Unknown tool" in msg


async def test_workspace_from_settings(tmp_path):
    client = _client(tmp_path)
    worker = WorkerAgent(_task(), client=client)
    assert worker.workspace == tmp_path
