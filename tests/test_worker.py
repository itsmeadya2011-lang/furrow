from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from furrow.agents.prompts import WORKER_TOOLS
from furrow.agents.worker import WorkerAgent, is_command_denied, resolve_in_workspace
from furrow.config import Settings
from furrow.llm import LLMResponse, ToolCall


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools=None, system="", model=None):
        self.calls.append({"messages": messages, "tools": tools, "system": system, "model": model})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def complete(self, prompt, system="", model=None):
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

    async def chat(self, messages, tools=None, system="", model=None):
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
    settings = make_settings(tmp_path).model_copy(update={"command_timeout": 0.2})
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
    settings = make_settings(tmp_path).model_copy(update={"command_timeout": 0.2})
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
    assert sorted(names) == ["list_files", "read_file", "run_command", "write_file"]
    assert len(WORKER_TOOLS) == 4
    for t in WORKER_TOOLS:
        assert "name" in t and "description" in t and "parameters" in t
        p = t["parameters"]
        assert p["type"] == "object"
        assert "properties" in p
        assert "required" in p


def test_resolve_in_workspace():
    ws = Path("/tmp/ws").resolve()
    assert resolve_in_workspace(ws, "../x") is None
    assert resolve_in_workspace(ws, "/etc/passwd") is None
    inside = resolve_in_workspace(ws, str(ws / "a.py"))
    assert inside == ws / "a.py"


@pytest.mark.asyncio
async def test_denied_command_table(tmp_path, tmp_path_factory, monkeypatch):
    """Table-driven test with commands that MUST be denied.

    Includes the two cases the legacy substring check used to cover: the fork
    bomb (destroyed by the segment splitter, so it is matched against the
    whole command) and ``chmod -R 777 /`` (matched case-insensitively).

    The rows that mention ``$HOME``/``~`` must never touch the real home
    directory, so HOME points at a throwaway directory that lives *outside*
    the workspace. If it lived inside, a redirect such as ``> ~/x`` would
    stay inside the workspace and would legitimately be allowed.
    """
    denied_commands = [
        "/bin/rm -rf /",
        "rm  -rf  /",
        "rm -rf /*",
        "echo hi; rm -rf /",
        "rm -fr /",
        ":(){:|:&};:",
        "mkfs.ext4 /dev/sda",
        "dd if=/dev/zero of=/dev/sda",
        "shutdown -h now",
        "git push --force",
        "git push -f origin main",
        "echo hacked > /etc/passwd",
        "chmod -R 777 /",
        "chmod -r 777 /",
        # command substitutions hide the dangerous part of the command
        "echo $(rm -rf /)",
        "echo `rm -rf /`",
        # every spelling of the HOME variable
        "rm -rf \"$HOME\"",
        "rm -rf ${HOME}",
        "rm -rf '$HOME'",
        # redirections that escape the workspace
        "echo hi > ../../etc/passwd",
        "echo hi > ~/x",
        "echo hi > $HOME/x",
        "echo hi > ${HOME}/x",
    ]
    from furrow.config import TaskModel

    # Throwaway home directory, deliberately outside of the workspace.
    fake_home = tmp_path_factory.mktemp("fake_home")
    assert not fake_home.is_relative_to(tmp_path)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))

    for cmd in denied_commands:
        marker = tmp_path / f"marker_{hash(cmd)}.txt"
        # Use command that would create marker if executed
        test_cmd = cmd
        # For commands that might not create marker, the denial is what matters
        responses = [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="t1", name="run_command", arguments={"command": test_cmd})
                ],
            ),
            LLMResponse(text="done", tool_calls=[]),
        ]
        client = FakeLLMClient(responses)
        settings = make_settings(tmp_path)
        task = TaskModel(id="1", description="cmd")
        agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
        await agent.run()
        second_call = client.calls[1]["messages"]
        found_denied = False
        for m in second_call:
            if isinstance(m.get("content"), list):
                for b in m["content"]:
                    if b.get("type") == "tool_result":
                        assert b["content"] == "error: command denied", (
                            f"Command '{cmd}' not denied properly"
                        )
                        found_denied = True
        assert found_denied
        assert not marker.exists()

    # None of the commands was executed: the throwaway home directory is
    # still there and still empty.
    assert fake_home.exists()
    assert not any(fake_home.iterdir())


@pytest.mark.asyncio
async def test_allowed_command_table(tmp_path):
    """Table-driven test with commands that MUST still be allowed."""
    allowed_commands = [
        "pytest -q",
        "python -m ruff check .",
        "git status",
        "rm -rf build",
        "ls -la",
        "npm test --silent",
        "cargo test -q",
        "echo hi > out.txt",
    ]
    from furrow.config import TaskModel

    for cmd in allowed_commands:
        responses = [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="t1", name="run_command", arguments={"command": cmd})
                ],
            ),
            LLMResponse(text="done", tool_calls=[]),
        ]
        client = FakeLLMClient(responses)
        settings = make_settings(tmp_path)
        task = TaskModel(id="1", description="cmd")
        agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
        await agent.run()
        second_call = client.calls[1]["messages"]
        found_result = False
        for m in second_call:
            if isinstance(m.get("content"), list):
                for b in m["content"]:
                    if b.get("type") == "tool_result":
                        result = b["content"]
                        assert result != "error: command denied", (
                            f"Command '{cmd}' was incorrectly denied"
                        )
                        found_result = True
        assert found_result


def test_is_command_denied_guard_level(tmp_path, tmp_path_factory, monkeypatch):
    """Guard-level policy check, independent of the command executor.

    Denied commands must be rejected and ordinary commands must stay
    allowed. HOME points at a throwaway directory outside of the workspace so
    that a regression can never reach the real home directory.
    """
    fake_home = tmp_path_factory.mktemp("fake_home")
    assert not fake_home.is_relative_to(tmp_path)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))

    denied_commands = [
        "rm -rf /",
        "rm -rf ~",
        'rm -rf "$HOME"',
        "rm -rf ${HOME}",
        "rm -rf '$HOME'",
        "mkfs.ext4 /dev/sda",
        "dd if=/dev/zero of=/dev/sda",
        "fdisk /dev/sda",
        "shutdown -h now",
        "git push --force",
        "git push -f origin main",
        "echo hacked > /etc/passwd",
        "echo hi > ../../etc/passwd",
        "echo hi > ~/x",
        "echo hi > $HOME/x",
        "echo hi > ${HOME}/x",
    ]
    allowed_commands = [
        "pytest -q",
        "git status",
        "ls -f",
        "rm -rf build",
        "echo hi > out.txt",
        "cd sub && ls",
    ]

    for cmd in denied_commands:
        assert is_command_denied(cmd, tmp_path) is True, f"'{cmd}' should be denied"
    for cmd in allowed_commands:
        assert is_command_denied(cmd, tmp_path) is False, f"'{cmd}' should be allowed"


def test_is_command_denied_checks_command_substitutions(tmp_path):
    """The text of a command substitution is checked with the same rules.

    ``fdisk`` is denied by program name, so the substitution
    ``$(fdisk /dev/sda)`` has to be denied as well - and this only works
    when the substitution itself is inspected.
    """
    assert is_command_denied("echo $(fdisk /dev/sda)", tmp_path) is True
    assert is_command_denied("echo `fdisk /dev/sda`", tmp_path) is True


def test_is_command_denied_allows_redirect_inside_workspace(tmp_path):
    """A redirect whose target stays inside the workspace is allowed."""
    assert is_command_denied("echo hi > out.txt", tmp_path) is False


@pytest.mark.asyncio
async def test_shell_and_chaining(tmp_path):
    from furrow.config import TaskModel

    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="run_command",
                    arguments={"command": "echo a && echo b"},
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    second_call = client.calls[1]["messages"]
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    content = b["content"]
                    assert "a" in content and "b" in content
                    found = True
    assert found


@pytest.mark.asyncio
async def test_shell_cd_changes_directory(tmp_path):
    from furrow.config import TaskModel

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    command = f"cd {subdir} && pwd"
    responses = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="t1", name="run_command", arguments={"command": command})],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    second_call = client.calls[1]["messages"]
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    content = b["content"]
                    assert str(subdir.resolve()) in content or "subdir" in content
                    found = True
    assert found


@pytest.mark.asyncio
async def test_shell_pipeline(tmp_path):
    from furrow.config import TaskModel

    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="run_command",
                    arguments={"command": "echo hi | cat"},
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    second_call = client.calls[1]["messages"]
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    content = b["content"]
                    assert "hi" in content
                    found = True
    assert found


@pytest.mark.asyncio
async def test_shell_env_assignment(tmp_path):
    from furrow.config import TaskModel

    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="run_command",
                    arguments={"command": "FOO=bar; echo $FOO"},
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    second_call = client.calls[1]["messages"]
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    content = b["content"]
                    assert "bar" in content
                    found = True
    assert found


@pytest.mark.asyncio
async def test_shell_redirection_creates_file(tmp_path):
    from furrow.config import TaskModel

    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="run_command",
                    arguments={"command": "echo hi > out.txt"},
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    await agent.run()
    assert (tmp_path / "out.txt").exists()
    assert (tmp_path / "out.txt").read_text().strip() == "hi"


@pytest.mark.asyncio
async def test_failing_command_does_not_raise(tmp_path):
    from furrow.config import TaskModel

    responses = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="t1", name="run_command", arguments={"command": "exit 3"})],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
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
                    # failing command returns output (possibly empty) and does not raise
                    assert b["content"] is not None
                    found = True
    assert found


@pytest.mark.asyncio
async def test_command_timeout_from_command_timeout_not_request_timeout(tmp_path):
    from furrow.config import TaskModel

    # request_timeout small, command_timeout large - sleep should NOT timeout
    settings = make_settings(tmp_path).model_copy(
        update={"request_timeout": 1.0, "command_timeout": 30.0}
    )
    assert float(settings.command_timeout) == 30.0
    agent = WorkerAgent(
        task=TaskModel(id="1", description="cmd"),
        client=FakeLLMClient([]),
        settings=settings,
        workspace=tmp_path,
    )
    out = await agent._run_command("sleep 2")
    assert "timed out" not in out.lower()


@pytest.mark.asyncio
async def test_kill_process_tree_waits_after_kill(tmp_path, monkeypatch):
    """_kill_process_tree reaps the process after killing it via the process group."""
    from furrow.agents.worker import WorkerAgent

    killed = {"pg": False, "kill": False}
    waited = False

    class FakeProc:
        pid = 4242

        def kill(self):
            killed["kill"] = True

        async def wait(self):
            nonlocal waited
            waited = True

    def fake_getpgid(pid):
        return 12345

    def fake_killpg(pgid, sig):
        killed["pg"] = True

    monkeypatch.setattr("os.getpgid", fake_getpgid)
    monkeypatch.setattr("os.killpg", fake_killpg)

    proc = FakeProc()
    await WorkerAgent._kill_process_tree(proc, new_session=True)

    assert killed["pg"] is True
    assert killed["kill"] is False
    assert waited is True


@pytest.mark.asyncio
async def test_kill_process_tree_fallback_kill_then_wait(tmp_path, monkeypatch):
    """When killpg is unavailable, fall back to proc.kill() then wait."""
    from furrow.agents.worker import WorkerAgent

    killed = {"pg": False, "kill": False}
    waited = False

    class FakeProc:
        pid = 4242

        def kill(self):
            killed["kill"] = True

        async def wait(self):
            nonlocal waited
            waited = True

    def fake_getpgid(pid):
        raise OSError("no pg")

    monkeypatch.setattr("os.getpgid", fake_getpgid)

    proc = FakeProc()
    await WorkerAgent._kill_process_tree(proc, new_session=True)

    assert killed["pg"] is False
    assert killed["kill"] is True
    assert waited is True


@pytest.mark.asyncio
async def test_kill_process_tree_no_session_uses_kill_then_wait(tmp_path):
    """When new_session is False, proc.kill() is called then wait."""
    from furrow.agents.worker import WorkerAgent

    killed = {"kill": False}
    waited = False

    class FakeProc:
        pid = 4242

        def kill(self):
            killed["kill"] = True

        async def wait(self):
            nonlocal waited
            waited = True

    proc = FakeProc()
    await WorkerAgent._kill_process_tree(proc, new_session=False)

    assert killed["kill"] is True
    assert waited is True
