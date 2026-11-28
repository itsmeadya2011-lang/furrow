from __future__ import annotations

import asyncio
import os
import signal
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


@pytest.mark.asyncio
async def test_run_command_cancelled_kills_process_group(tmp_path, monkeypatch):
    """Cancelling a run_command turn kills the whole process group and lets
    asyncio.CancelledError propagate (worker.run() does not convert it into a
    failed WorkerResult)."""
    from furrow.config import TaskModel

    killpg_calls = []
    getpgid_calls = []

    def fake_getpgid(pid):
        getpgid_calls.append(pid)
        return 1234

    def fake_killpg(pgid, sig):
        killpg_calls.append((pgid, sig))

    monkeypatch.setattr(os, "getpgid", fake_getpgid)
    monkeypatch.setattr(os, "killpg", fake_killpg)

    spawned = asyncio.Event()
    communicating = asyncio.Event()
    blocker = asyncio.Event()  # keep communicate() blocked until released

    class FakeProc:
        def __init__(self, **kwargs):
            self.pid = 999
            self.returncode = None
            self.kwargs = kwargs
            spawned.set()

        def kill(self):
            self.killed = True

        async def wait(self):
            return 0

        async def communicate(self):
            communicating.set()
            await blocker.wait()
            return (b"", b"")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProc(**kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(id="t1", name="run_command", arguments={"command": "sleep 100"})
            ],
        ),
        # Unreachable: the task is cancelled before the second LLM call.
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)

    task_obj = asyncio.create_task(agent.run())
    await spawned.wait()
    await communicating.wait()  # communicate() is in flight
    task_obj.cancel()
    cancelled = False
    try:
        await task_obj
    except asyncio.CancelledError:
        cancelled = True
    assert cancelled is True, "asyncio.CancelledError must propagate out of worker.run()"
    assert getpgid_calls == [999], getpgid_calls
    assert killpg_calls == [(1234, signal.SIGKILL)], killpg_calls


@pytest.mark.asyncio
async def test_run_command_returns_output(tmp_path, monkeypatch):
    """Regression: a normal successful run_command returns its output unchanged."""
    from furrow.config import TaskModel

    class FakeProc:
        def __init__(self, **kwargs):
            self.returncode = 0

        async def communicate(self):
            return (b"hello output\n", b"")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProc(**kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=FakeLLMClient([]), settings=settings, workspace=tmp_path)
    out = await agent._run_command("echo hi")
    assert out == "hello output\n"


@pytest.mark.asyncio
async def test_list_files_caps_at_500_entries(tmp_path):
    for i in range(520):
        (tmp_path / f"f{i:04d}.txt").write_text("x")
    client = FakeLLMClient([])
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="list")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    out = await agent._list_files(".")
    lines = out.splitlines()
    # 500 file lines plus one note telling the model 20 entries were omitted.
    assert len(lines) == 501
    assert "20 more entries not shown" in lines[-1]


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


def test_new_command_guard_bypasses(tmp_path):
    """Guard-level tests for newly closed command-guard bypasses."""
    denied_commands = [
        "echo hi >| /etc/passwd",
        "echo hi > $(cat /tmp/x)",
        "echo hi > $OUTSIDE",
        "echo hi | tee /etc/passwd",
        "git push origin +main",
        "git push origin HEAD:+main",
        "echo hi & fdisk /dev/sda",
    ]
    allowed_commands = [
        "echo hi | tee out.txt",
        "git push origin main",
        "echo hi &",
        # noclobber redirect inside workspace is allowed
        "echo hi >| out.txt",
        # || operator still splits and is harmless
        "echo a || echo b",
        # ordinary pipe still allowed
        "echo hi | cat",
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


def test_is_command_denied_checks_process_substitutions(tmp_path):
    """The text of a process substitution is checked with the same rules.

    Bash process substitution ``<(...)`` and ``>(...)`` runs an arbitrary
    command, so its inner text has to be denied just like a command
    substitution - including nested process substitutions.
    """
    denied_commands = [
        "cat <(rm -rf /)",
        "diff <(echo a) <(fdisk /dev/sda)",
        "cat <(echo <(mkfs.ext4 /dev/sda))",
        "tee >(chmod -R 777 /)",
    ]
    allowed_commands = [
        "diff <(sort a.txt) <(sort b.txt)",
        "cat <(echo hi)",
        "echo hi > out.txt",
        "echo $(echo ok)",
    ]
    for cmd in denied_commands:
        assert is_command_denied(cmd, tmp_path) is True, f"'{cmd}' should be denied"
    for cmd in allowed_commands:
        assert is_command_denied(cmd, tmp_path) is False, f"'{cmd}' should be allowed"


def test_extract_command_substitutions_includes_process_substitutions():
    """``_extract_command_substitutions`` returns process substitution text."""
    from furrow.agents.worker import _extract_command_substitutions

    substitutions = _extract_command_substitutions("diff <(sort a) >(cat)")
    assert "sort a" in substitutions
    assert "cat" in substitutions


def test_is_command_denied_allows_redirect_inside_workspace(tmp_path):
    """A redirect whose target stays inside the workspace is allowed."""
    assert is_command_denied("echo hi > out.txt", tmp_path) is False


def test_is_command_denied_rm_token_aware(tmp_path):
    """Dangerous ``rm`` is caught token-aware regardless of flag order."""
    denied_commands = [
        "rm -r -f /",
        "rm --recursive --force /",
        "rm -Rf /",
        "rm -rf -- /",
        "sudo rm -rf /",
    ]
    allowed_commands = [
        "rm -rf build",
        f"rm -rf {tmp_path}/build",
        # no targets at all: nothing to recursively force-delete
        "rm -rf --",
    ]
    for cmd in denied_commands:
        assert is_command_denied(cmd, tmp_path) is True, f"'{cmd}' should be denied"
    for cmd in allowed_commands:
        assert is_command_denied(cmd, tmp_path) is False, f"'{cmd}' should be allowed"


def test_is_command_denied_chmod_token_aware(tmp_path):
    """Recursive ``chmod`` outside the workspace is caught token-aware."""
    denied_commands = [
        "chmod 777 -R /",
        "chmod -R 777 /*",
        "sudo chmod -R 777 /",
    ]
    allowed_commands = [
        "chmod -R 777 ./sub",
    ]
    for cmd in denied_commands:
        assert is_command_denied(cmd, tmp_path) is True, f"'{cmd}' should be denied"
    for cmd in allowed_commands:
        assert is_command_denied(cmd, tmp_path) is False, f"'{cmd}' should be allowed"


def test_is_command_denied_allows_special_write_targets(tmp_path):
    """Redirections and tee to /dev/null and friends are allowed."""
    allowed_commands = [
        "echo hi > /dev/null",
        "ls 2>/dev/null",
        "echo hi | tee /dev/null",
    ]
    for cmd in allowed_commands:
        assert is_command_denied(cmd, tmp_path) is False, f"'{cmd}' should be allowed"


def test_is_command_denied_wrapper_bypass(tmp_path):
    """Guard-level tests for wrapper program bypasses (sudo, command, env, nohup)."""
    denied_commands = [
        "sudo fdisk /dev/sda",
        "command mkfs.ext4 /dev/sda",
        "env FOO=1 shutdown -h now",
        "nohup parted /dev/sda",
        "doas fdisk /dev/sda",
        "exec fdisk /dev/sda",
        "sudo rm -rf /",
        "env chmod -R 777 /",
    ]
    allowed_commands = [
        "echo rm",
        "echo chmod",
        "git status",
        "ls -f",
    ]

    for cmd in denied_commands:
        assert is_command_denied(cmd, tmp_path) is True, f"'{cmd}' should be denied"
    for cmd in allowed_commands:
        assert is_command_denied(cmd, tmp_path) is False, f"'{cmd}' should be allowed"


def test_is_command_denied_wrapper_aware_tee_and_force_push(tmp_path):
    """Wrapper-aware write detection and force-push detection stay exact.

    ``tee`` writes through a path argument, so it must be resolved through the
    wrapper (``sudo``/``env``/``command``) to its real program and targets. A
    force push must only be denied for an actual ``git ... push`` carrying a
    force flag, not merely because ``push`` or ``-f`` appear somewhere.
    """
    denied_commands = [
        "sudo tee /etc/passwd",
        "env FOO=1 tee /etc/passwd",
        "command tee /etc/passwd",
        "sudo -n tee /etc/passwd",
        "git -C repo push --force",
    ]
    allowed_commands = [
        "echo hi | sudo tee out.txt",
        "echo hi | tee /dev/null",
        "grep -f push-notes git-log",
        "git status",
        "git push origin main",
    ]

    for cmd in denied_commands:
        assert is_command_denied(cmd, tmp_path) is True, f"'{cmd}' should be denied"
    for cmd in allowed_commands:
        assert is_command_denied(cmd, tmp_path) is False, f"'{cmd}' should be allowed"


def test_is_command_denied_shell_interpreter_and_cd_escapes(tmp_path):
    """A shell ``-c`` script is inspected and a ``cd`` outside the workspace is denied.

    ``bash -c 'rm -rf /'`` runs an arbitrary script that is not a separate
    shell segment, so the script text has to be checked recursively.
    ``cd / && echo x > etc/passwd`` changes the working directory before a
    relative redirection, so the ``cd`` target itself must stay in the
    workspace. ``eval``/``source`` run arbitrary text and are denied outright.
    """
    denied_commands = [
        "bash -c 'rm -rf /'",
        "sh -c 'echo hacked > /etc/passwd'",
        "sudo bash -c 'fdisk /dev/sda'",
        "zsh -lc 'chmod -R 777 /'",
        "cd / && echo x > etc/passwd",
        "cd .. && rm -rf build",
        "cd ~",
        "cd $HOME",
        "cd",
        "cd -",
        "eval 'rm -rf /'",
        "source /tmp/evil.sh",
    ]
    allowed_commands = [
        "bash -c 'echo hi'",
        "sh -c 'ls -la'",
        f"cd {tmp_path} && pwd",
        "cd sub && ls",
        "cd subdir && make",
    ]

    for cmd in denied_commands:
        assert is_command_denied(cmd, tmp_path) is True, f"'{cmd}' should be denied"
    for cmd in allowed_commands:
        assert is_command_denied(cmd, tmp_path) is False, f"'{cmd}' should be allowed"


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
async def test_write_file_missing_content(tmp_path):
    from furrow.config import TaskModel

    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="x")
    agent = WorkerAgent(task=task, client=FakeLLMClient([]), settings=settings, workspace=tmp_path)
    target = tmp_path / "keep.py"
    tc = ToolCall(id="t1", name="write_file", arguments={"path": str(target)})
    out = await agent._execute_tool(tc)
    assert out == "error: missing content"
    assert not target.exists()


@pytest.mark.asyncio
async def test_write_file_invalid_content(tmp_path):
    from furrow.config import TaskModel

    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="x")
    agent = WorkerAgent(task=task, client=FakeLLMClient([]), settings=settings, workspace=tmp_path)
    target = tmp_path / "keep.py"
    for content in (None, 123, ["a"]):
        tc = ToolCall(
            id="t1",
            name="write_file",
            arguments={"path": str(target), "content": content},
        )
        out = await agent._execute_tool(tc)
        assert out.startswith("error: ")
        assert "content" in out
    assert not target.exists()


@pytest.mark.asyncio
async def test_write_file_valid_content_writes_file(tmp_path):
    from furrow.config import TaskModel

    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="x")
    agent = WorkerAgent(task=task, client=FakeLLMClient([]), settings=settings, workspace=tmp_path)
    target = tmp_path / "keep.py"
    tc = ToolCall(
        id="t1",
        name="write_file",
        arguments={"path": str(target), "content": "print('hi')"},
    )
    out = await agent._execute_tool(tc)
    assert out.startswith("wrote ")
    assert target.read_text() == "print('hi')"


@pytest.mark.asyncio
async def test_run_command_exit_status_marker(tmp_path):
    from furrow.config import TaskModel

    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=FakeLLMClient([]), settings=settings, workspace=tmp_path)
    out = await agent._run_command("exit 3")
    assert "[exit status 3]" in out


@pytest.mark.asyncio
async def test_run_command_exit_status_zero_no_marker(tmp_path):
    from furrow.config import TaskModel

    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=FakeLLMClient([]), settings=settings, workspace=tmp_path)
    out = await agent._run_command("echo hi")
    assert "[exit status" not in out
    assert "hi" in out


@pytest.mark.asyncio
async def test_truncated_response_is_failure(tmp_path):
    from furrow.config import TaskModel

    responses = [LLMResponse(text="partial", tool_calls=[], stop_reason="max_tokens")]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="x")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    result = await agent.run()
    assert result.success is False
    assert "truncated" in (result.error or "")
    assert "max_tokens" in (result.error or "")


@pytest.mark.asyncio
async def test_end_turn_response_still_success(tmp_path):
    from furrow.config import TaskModel

    responses = [LLMResponse(text="done", tool_calls=[], stop_reason="end_turn")]
    client = FakeLLMClient(responses)
    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="x")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    result = await agent.run()
    assert result.success is True
    assert result.summary == "done"


@pytest.mark.asyncio
async def test_create_subprocess_exec_passes_stdin_devnull(tmp_path, monkeypatch):
    from furrow.config import TaskModel

    captured = {}

    class FakeProc:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.returncode = 0

        async def communicate(self):
            return (b"", b"")

    async def fake_create_subprocess_exec(*args, **kwargs):
        proc = FakeProc(**kwargs)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="cmd")
    agent = WorkerAgent(task=task, client=FakeLLMClient([]), settings=settings, workspace=tmp_path)
    out = await agent._run_command("echo hi")
    assert captured["stdin"] is asyncio.subprocess.DEVNULL
    assert out == ""


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
async def test_execute_tool_invalid_args_do_not_raise(tmp_path):
    from furrow.config import TaskModel

    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="x")
    agent = WorkerAgent(task=task, client=FakeLLMClient([]), settings=settings, workspace=tmp_path)
    for bad in ([1, 2], "not json", None):
        tc = ToolCall(id="t", name="read_file", arguments=bad)
        out = await agent._execute_tool(tc)
        assert isinstance(out, str)
        assert out.startswith("error: ")

def test_coerce_tool_args_deeply_nested_returns_none(monkeypatch):
    """Deeply nested JSON raises RecursionError inside json.loads; the worker

    must not let that propagate out of the turn - it must return None."""

    def _raise(_text):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr("furrow.agents.worker.json.loads", _raise)
    assert WorkerAgent._coerce_tool_args("[[1]]") is None


def test_coerce_tool_args_basic_behavior():
    """Normal dict / JSON-string / invalid-string behavior is preserved."""
    assert WorkerAgent._coerce_tool_args(None) == {}
    assert WorkerAgent._coerce_tool_args({}) == {}
    assert WorkerAgent._coerce_tool_args({"a": 1}) == {"a": 1}
    assert WorkerAgent._coerce_tool_args('{"a": 1}') == {"a": 1}
    assert WorkerAgent._coerce_tool_args("") == {}
    assert WorkerAgent._coerce_tool_args("   ") == {}
    assert WorkerAgent._coerce_tool_args("not json") is None
    assert WorkerAgent._coerce_tool_args("[1, 2]") is None
    assert WorkerAgent._coerce_tool_args(123) is None
    assert WorkerAgent._coerce_tool_args([1, 2]) is None


@pytest.mark.asyncio
async def test_read_file_truncation_marker(tmp_path):
    big = "a" * 25000
    (tmp_path / "big.txt").write_text(big)
    client = FakeLLMClient([])
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="read")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)
    out = await agent._read_file("big.txt")
    assert "[... output truncated ...]" in out


 
@pytest.mark.asyncio
async def test_run_command_stderr_newline_format(tmp_path):
    from furrow.config import TaskModel

    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(id="t1", name="run_command", arguments={"command": "echo err >&2"})
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
                    assert "\n[stderr]" in content
                    assert "\\n[stderr]" not in content
                    found = True
    assert found

    import threading

    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")

    client = FakeLLMClient([])
    settings = make_settings(tmp_path)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="list")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)

    loop_thread = threading.get_ident()
    seen = {}

    def slow_list(directory):
        seen["tid"] = threading.get_ident()
        import time
        time.sleep(0.1)
        from pathlib import Path as _P
        return [str(p) for p in _P(directory).rglob("*") if _P(p).is_file()]

    client.list_files = slow_list

    progress = {"n": 0}

    async def ticker():
        while not progress.get("done"):
            progress["n"] += 1
            await asyncio.sleep(0.01)

    t = asyncio.ensure_future(ticker())
    await agent._list_files(".")
    progress["done"] = True
    await t
    assert seen["tid"] != loop_thread
    assert progress["n"] > 0

@pytest.mark.skipif(os.name != "posix", reason="symlinks require POSIX")
def test_resolve_in_workspace_rejects_symlink_escape(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    link = ws / "link.txt"
    link.symlink_to(secret)
    assert resolve_in_workspace(ws, str(link)) is None


@pytest.mark.skipif(os.name != "posix", reason="symlinks require POSIX")
@pytest.mark.asyncio
async def test_write_file_through_absolute_symlink_is_rejected(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = ws / "link"
    link.symlink_to(outside)
    responses = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="write_file",
                    arguments={"path": str(ws / "link" / "evil.py"), "content": "x"},
                )
            ],
        ),
        LLMResponse(text="done", tool_calls=[]),
    ]
    client = FakeLLMClient(responses)
    settings = make_settings(ws)
    from furrow.config import TaskModel

    task = TaskModel(id="1", description="escape")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=ws)
    await agent.run()
    assert not (outside / "evil.py").exists()
    second_call = client.calls[1]["messages"]
    found = False
    for m in second_call:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    assert "escapes workspace" in b["content"]
                    found = True
    assert found


@pytest.mark.asyncio
async def test_kill_process_tree_fallback_reaps_child():
    """Regression: the fallback path awaits proc.wait() to reap the child."""

    class FakeProc:
        def __init__(self):
            self.pid = 4242
            self.killed = False
            self.waited = False

        def kill(self):
            self.killed = True

        async def wait(self):
            self.waited = True
            return 0

    fake = FakeProc()
    await WorkerAgent._kill_process_tree(fake, False)
    assert fake.killed is True
    assert fake.waited is True


@pytest.mark.asyncio
async def test_list_files_reports_omitted_entries(tmp_path, monkeypatch):
    """When the listing is capped, a note reports how many entries were omitted."""
    from furrow.config import TaskModel

    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")
    monkeypatch.setattr("furrow.agents.worker.MAX_LIST_ENTRIES", 3)
    client = FakeLLMClient([])
    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="list")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)

    out = await agent._list_files(".")

    lines = out.splitlines()
    assert len(lines) == 4  # 3 file lines + 1 note
    assert "2 more entries not shown" in lines[-1]


@pytest.mark.asyncio
async def test_list_files_no_note_when_under_cap(tmp_path):
    """A listing that fits under the cap has no truncation note."""
    from furrow.config import TaskModel

    (tmp_path / "a.txt").write_text("x")
    client = FakeLLMClient([])
    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="list")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)

    out = await agent._list_files(".")

    assert "not shown" not in out
    assert "a.txt" in out


@pytest.mark.asyncio
async def test_read_file_reports_total_size(tmp_path):
    """A truncated read reports the file's total size, not just the marker."""
    from furrow.config import TaskModel

    big = "a" * 25000
    (tmp_path / "big.txt").write_text(big)
    client = FakeLLMClient([])
    settings = make_settings(tmp_path)
    task = TaskModel(id="1", description="read")
    agent = WorkerAgent(task=task, client=client, settings=settings, workspace=tmp_path)

    out = await agent._read_file("big.txt")

    assert "[... output truncated ...]" in out
    assert "25000 chars total" in out
