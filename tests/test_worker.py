from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from furrow.agents.prompts import WORKER_TOOLS
from furrow.agents.worker import WorkerAgent, resolve_in_workspace
from furrow.config import Settings
from furrow.llm import LLMResponse, ToolCall


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools=None, system="", model=None, role="default"):
        self.calls.append({"messages": messages, "tools": tools, "system": system, "model": model})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def complete(self, prompt, system="", model=None, role="default"):
        return ""

    async def read_file(self, path):
        return Path(path).read_text()

    async def write_file(self, path, content):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)

    def list_files(self, directory):
        return [str(p) for p in Path(directory).rglob("*") if p.is_file()]


class LoopingLLMClient(FakeLLMClient):
    """Fake client that always answers with the same tool-calling response."""

    def __init__(self, response):
        super().__init__([response])
        self.response = response

    async def chat(self, messages, tools=None, system="", model=None, role="default"):
        self.calls.append({"messages": messages, "tools": tools, "system": system, "model": model})
        return self.response


def make_settings(tmp_path, iters=12):
    return Settings(workspace=tmp_path, max_worker_iterations=iters)


@pytest.mark.asyncio
async def test_write_file_then_finish(tmp_path):
    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="write_file",
                    arguments={"path": "hello.py", "content": "print('hi')"},
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    task = __import__("furrow.config", fromlist=["TaskModel"]).TaskModel(
        id="1", description="write hello", files=["hello.py"]
    )
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    result = await agent.run()
    assert result.success is True
    assert result.files_changed == ["hello.py"]
    assert (tmp_path / "hello.py").read_text() == "print('hi')"


@pytest.mark.asyncio
async def test_immediate_text(tmp_path):
    responses = [LLMResponse(text="all good", tool_calls=[])]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="noop")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    result = await agent.run()
    assert result.success is True
    assert result.summary == "all good"
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_read_file(tmp_path):
    (tmp_path / "a.txt").write_text("content here")
    responses = [
        LLMResponse(
            text="", tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "a.txt"})]
        ),
        LLMResponse(text="read", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="read")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    # check tool result message
    second_call = client.calls[1]["messages"]
    # find tool_result
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    assert "content here" in b["content"]
                    found = True
    assert found


@pytest.mark.asyncio
async def test_read_missing_file(tmp_path):
    responses = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "missing.txt"})],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="read")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    second_call = client.calls[1]["messages"]
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    assert "error" in b["content"].lower()
                    found = True
    assert found


@pytest.mark.asyncio
async def test_path_escape_write(tmp_path):
    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1", name="write_file", arguments={"path": "../evil.py", "content": "x"}
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="escape")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    assert not (tmp_path.parent / "evil.py").exists()


@pytest.mark.asyncio
async def test_denied_command(tmp_path):
    responses = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="t1", name="run_command", arguments={"command": "rm -rf /"})],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    second_call = client.calls[1]["messages"]
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    assert "denied" in b["content"].lower()


@pytest.mark.asyncio
async def test_safe_command(tmp_path):
    responses = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="t1", name="run_command", arguments={"command": "echo hi"})],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    second_call = client.calls[1]["messages"]
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    assert "hi" in b["content"]


@pytest.mark.asyncio
async def test_iteration_budget(tmp_path):
    responses = [
        LLMResponse(
            text="", tool_calls=[ToolCall(id=f"t{i}", name="read_file", arguments={"path": "x"})]
        )
        for i in range(5)
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path, iters=3)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="loop")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    result = await agent.run()
    assert result.iterations == 3
    assert result.error == "worker iteration budget exhausted"
    # A truncated worker must never look successful, otherwise the orchestrator
    # marks the task as completed and never retries it.
    assert result.success is False


@pytest.mark.asyncio
async def test_iteration_budget_exhausted_reports_failure(tmp_path):
    """Regression: exhausting the iteration budget yields success=False + error."""
    response = LLMResponse(
        text="", tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "x"})]
    )
    client = LoopingLLMClient(response)
    settings = make_settings(tmp_path).model_copy(update={"max_worker_iterations": 2})
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="loop")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    result = await agent.run()
    assert result.success is False
    assert result.error
    assert result.iterations == 2
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_run_command_timeout(tmp_path):
    """Regression: a slow command times out and returns the timeout message."""
    responses = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="t1", name="run_command", arguments={"command": "sleep 5"})],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path).model_copy(update={"request_timeout": 0.2})
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    result = await agent.run()
    assert result.success is True
    second_call = client.calls[1]["messages"]
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    assert "timed out" in b["content"]
                    found = True
    assert found


@pytest.mark.asyncio
async def test_run_command_timeout_kills_grandchildren(tmp_path):
    """Regression: timing out kills the whole process group, leaving no orphans."""
    marker = tmp_path / "orphan.txt"
    # The subshell is a grandchild of the worker: killing only the direct child
    # leaves it alive and it would create the marker file.
    command = f"sh -c '(sleep 0.5; touch {marker}) & wait'"
    settings = make_settings(tmp_path).model_copy(update={"request_timeout": 0.2})
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=FakeLLMClient([]), settings=settings, workspace=tmp_path)
    out = await agent._run_command(command)
    assert out == "command timed out"
    await asyncio.sleep(1.0)
    assert not marker.exists()


def test_list_files_caps_at_500_entries(tmp_path):
    for i in range(520):
        (tmp_path / f"f{i:04d}.txt").write_text("x")
    client = FakeLLMClient([])
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="list")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    out = agent._list_files(".")
    lines = out.splitlines()
    assert len(lines) == 500


@pytest.mark.asyncio
async def test_client_exception(tmp_path):
    client = FakeLLMClient([RuntimeError("boom")])
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="err")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    result = await agent.run()
    assert result.success is False
    assert "boom" in (result.error or "")


@pytest.mark.asyncio
async def test_assistant_turn_blocks(tmp_path):
    responses = [
        LLMResponse(
            text="thinking",
            tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "x"})],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="x")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    # second call messages should contain assistant turn with tool_use
    second = client.calls[1]["messages"]
    assistant_msgs = [m for m in second if m["role"] == "assistant"]
    assert assistant_msgs
    blocks = assistant_msgs[0]["content"]
    types = [b["type"] for b in blocks]
    assert "tool_use" in types
    # tool results as user message
    user_msgs = [
        m
        for m in second
        if m["role"] == "user"
        and isinstance(m["content"], list)
        and m["content"]
        and m["content"][0].get("type") == "tool_result"
    ]
    assert user_msgs


def test_worker_tools():
    names = [t["name"] for t in WORKER_TOOLS]
    assert sorted(names) == [
        "edit_file",
        "grep_search",
        "list_files",
        "read_file",
        "run_command",
        "write_file",
    ]
    assert len(WORKER_TOOLS) == 6
    for t in WORKER_TOOLS:
        assert "name" in t and "description" in t and "parameters" in t
        p = t["parameters"]
        assert p["type"] == "object"
        assert "properties" in p
        assert "required" in p


@pytest.mark.asyncio
async def test_edit_file_replaces_substring(tmp_path):
    (tmp_path / "a.txt").write_text("hello world\nfoo bar")
    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="edit_file",
                    arguments={
                        "path": "a.txt",
                        "old_text": "hello world",
                        "new_text": "goodbye world",
                    },
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="edit")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    result = await agent.run()
    assert result.success is True
    assert (tmp_path / "a.txt").read_text() == "goodbye world\nfoo bar"


@pytest.mark.asyncio
async def test_edit_file_old_text_not_found(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="edit_file",
                    arguments={
                        "path": "a.txt",
                        "old_text": "nonexistent",
                        "new_text": "x",
                    },
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="edit")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    second_call = client.calls[1]["messages"]
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    assert "not found" in b["content"].lower()
                    found = True
    assert found


@pytest.mark.asyncio
async def test_edit_file_ambiguous_old_text(tmp_path):
    (tmp_path / "a.txt").write_text("foo\nfoo\nbar")
    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="edit_file",
                    arguments={
                        "path": "a.txt",
                        "old_text": "foo",
                        "new_text": "baz",
                    },
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="edit")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    second_call = client.calls[1]["messages"]
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    assert "ambiguous" in b["content"].lower()
                    found = True
    assert found


@pytest.mark.asyncio
async def test_grep_search_finds_matches(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("def bar():\n    return foo()\n")
    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="grep_search",
                    arguments={"pattern": "def foo"},
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="grep")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    second_call = client.calls[1]["messages"]
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    assert "a.py" in b["content"]
                    assert "def foo" in b["content"]
                    found = True
    assert found


@pytest.mark.asyncio
async def test_grep_search_no_matches(tmp_path):
    (tmp_path / "a.py").write_text("print('hi')\n")
    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="grep_search",
                    arguments={"pattern": "nonexistent_pattern_xyz"},
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="grep")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    second_call = client.calls[1]["messages"]
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    assert "no matches" in b["content"].lower()
                    found = True
    assert found


@pytest.mark.asyncio
async def test_grep_search_invalid_regex(tmp_path):
    (tmp_path / "a.py").write_text("x\n")
    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="grep_search",
                    arguments={"pattern": "[invalid"},
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="grep")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    second_call = client.calls[1]["messages"]
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    assert "invalid regex" in b["content"].lower()
                    found = True
    assert found


@pytest.mark.asyncio
async def test_write_file_size_limit(tmp_path):
    """write_file rejects content larger than MAX_WRITE_SIZE."""
    from furrow.agents.worker import MAX_WRITE_SIZE

    big_content = "x" * (MAX_WRITE_SIZE + 1)
    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="write_file",
                    arguments={"path": "big.txt", "content": big_content},
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="big")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    second_call = client.calls[1]["messages"]
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    assert "too large" in b["content"].lower()
                    found = True
    assert found
    assert not (tmp_path / "big.txt").exists()


@pytest.mark.asyncio
async def test_edit_file_path_escape(tmp_path):
    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="edit_file",
                    arguments={"path": "../evil.txt", "old_text": "x", "new_text": "y"},
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="escape")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    assert not (tmp_path.parent / "evil.txt").exists()


def test_resolve_in_workspace():
    ws = Path("/tmp/ws").resolve()
    assert resolve_in_workspace(ws, "../x") is None
    assert resolve_in_workspace(ws, "/etc/passwd") is None
    inside = resolve_in_workspace(ws, str(ws / "a.py"))
    assert inside == ws / "a.py"
