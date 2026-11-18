from __future__ import annotations

import asyncio
import os
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


# --- Tests for uncovered utility functions in worker.py ---

def test_normalize_command_edge_cases():
    """Test _normalize_command with various edge cases."""
    from furrow.agents.worker import _normalize_command

    # Empty command
    assert _normalize_command("") == ""
    assert _normalize_command("   ") == ""
    assert _normalize_command("\t\n") == ""

    # Surrounding quotes
    assert _normalize_command('"echo hello"') == "echo hello"
    assert _normalize_command("'echo hello'") == "echo hello"
    assert _normalize_command('"echo hello') == "echo hello"  # unbalanced
    assert _normalize_command('" "') == ""

    # HOME variable canonicalization
    assert _normalize_command('echo "$HOME"') == 'echo $home'
    assert _normalize_command("echo ${HOME}") == "echo $home"
    assert _normalize_command("echo $HOME") == "echo $home"
    assert _normalize_command("echo $home") == "echo $home"
    assert _normalize_command("echo ${home}") == "echo $home"
    assert _normalize_command('echo "$home"') == 'echo $home'
    assert _normalize_command("echo '$HOME'") == "echo $home"

    # Whitespace collapsing
    assert _normalize_command("echo    hello") == "echo hello"
    assert _normalize_command("  echo  hello  ") == "echo hello"
    assert _normalize_command("\techo\thello\n") == "echo hello"

    # Lowercasing
    assert _normalize_command("ECHO HELLO") == "echo hello"
    assert _normalize_command('Echo "$HOME"') == 'echo $home'


def test_split_command_segments_edge_cases():
    """Test _split_command_segments with various edge cases."""
    from furrow.agents.worker import _split_command_segments

    # Empty command
    assert _split_command_segments("") == []
    assert _split_command_segments("   ") == []
    assert _split_command_segments("\n\n") == []

    # Basic separators
    assert _split_command_segments("echo a; echo b") == ["echo a", "echo b"]
    assert _split_command_segments("echo a && echo b") == ["echo a", "echo b"]
    assert _split_command_segments("echo a || echo b") == ["echo a", "echo b"]
    assert _split_command_segments("echo a | echo b") == ["echo a", "echo b"]
    assert _split_command_segments("echo a & echo b") == ["echo a", "echo b"]
    assert _split_command_segments("echo a\necho b") == ["echo a", "echo b"]

    # Multiple separators
    assert _split_command_segments("a; b; c") == ["a", "b", "c"]
    assert _split_command_segments("a&&b||c") == ["a", "b", "c"]

    # Edge case: trailing separator
    assert _split_command_segments("echo a;") == ["echo a"]


def test_shell_tokens_edge_cases():
    """Test _shell_tokens with various edge cases."""
    from furrow.agents.worker import _shell_tokens

    # Normal cases
    assert _shell_tokens("echo hello world") == ["echo", "hello", "world"]
    assert _shell_tokens('echo "hello world"') == ["echo", "hello world"]
    assert _shell_tokens("echo 'hello world'") == ["echo", "hello world"]

    # Unbalanced quotes falls back to whitespace split
    assert _shell_tokens('echo "hello') == ['echo', '"hello']
    assert _shell_tokens("echo 'hello") == ["echo", "'hello"]

    # Empty
    assert _shell_tokens("") == []
    assert _shell_tokens("   ") == []


def test_get_program_basename_edge_cases():
    """Test _get_program_basename with various edge cases."""
    from furrow.agents.worker import _get_program_basename

    assert _get_program_basename("echo hello") == "echo"
    assert _get_program_basename("/usr/bin/echo hello") == "echo"
    assert _get_program_basename("./script.sh") == "script.sh"
    assert _get_program_basename("  echo  ") == "echo"
    assert _get_program_basename("") is None
    assert _get_program_basename("   ") is None
    assert _get_program_basename('"echo" hello') == "echo"  # quoted program name


def test_program_and_args_edge_cases():
    """Test _program_and_args with various edge cases."""
    from furrow.agents.worker import _program_and_args

    # Normal cases
    assert _program_and_args("echo hello") == ("echo", ["hello"])
    assert _program_and_args("sudo echo hello") == ("echo", ["hello"])
    assert _program_and_args("env FOO=bar echo hello") == ("echo", ["hello"])
    assert _program_and_args("nohup echo hello") == ("echo", ["hello"])
    assert _program_and_args("command echo hello") == ("echo", ["hello"])
    assert _program_and_args("exec echo hello") == ("echo", ["hello"])
    assert _program_and_args("doas echo hello") == ("echo", ["hello"])

    # Variable assignments
    assert _program_and_args("FOO=bar echo hello") == ("echo", ["hello"])
    assert _program_and_args("FOO=bar BAZ=qux echo hello") == ("echo", ["hello"])

    # Mixed
    assert _program_and_args("sudo FOO=bar echo hello") == ("echo", ["hello"])
    assert _program_and_args("env FOO=bar sudo echo hello") == ("echo", ["hello"])

    # Edge cases
    assert _program_and_args("") is None
    assert _program_and_args("   ") is None
    assert _program_and_args("FOO=bar") is None  # only assignment
    assert _program_and_args("sudo") is None  # only wrapper
    assert _program_and_args("sudo env") is None  # only wrappers


def test_is_short_flag_cluster_edge_cases():
    """Test _is_short_flag_cluster with various edge cases."""
    from furrow.agents.worker import _is_short_flag_cluster

    # Valid clusters
    assert _is_short_flag_cluster("-rf") is True
    assert _is_short_flag_cluster("-r") is True
    assert _is_short_flag_cluster("-abc") is True
    assert _is_short_flag_cluster("-rR") is True

    # Invalid clusters
    assert _is_short_flag_cluster("--force") is False  # long flag
    assert _is_short_flag_cluster("-") is False  # just dash
    assert _is_short_flag_cluster("--") is False  # double dash
    assert _is_short_flag_cluster("") is False
    assert _is_short_flag_cluster("echo") is False
    assert _is_short_flag_cluster("-1") is False  # no letters
    assert _is_short_flag_cluster("-=") is False  # no letters


def test_short_cluster_has_edge_cases():
    """Test _short_cluster_has with various edge cases."""
    from furrow.agents.worker import _short_cluster_has

    assert _short_cluster_has("-rf", "rf") is True
    assert _short_cluster_has("-rf", "r") is True
    assert _short_cluster_has("-rf", "f") is True
    assert _short_cluster_has("-rf", "x") is False
    assert _short_cluster_has("-rfR", "r") is True
    assert _short_cluster_has("--force", "f") is False  # not a cluster
    assert _short_cluster_has("-", "f") is False
    assert _short_cluster_has("", "f") is False


def test_is_dangerous_rm_edge_cases():
    """Test _is_dangerous_rm with various edge cases."""
    from pathlib import Path

    from furrow.agents.worker import _is_dangerous_rm

    ws = Path("/tmp/ws").resolve()

    # Not recursive + force
    assert _is_dangerous_rm(["-r", "/tmp/ws/file"], ws) is False
    assert _is_dangerous_rm(["-f", "/tmp/ws/file"], ws) is False
    assert _is_dangerous_rm(["--recursive", "/tmp/ws/file"], ws) is False
    assert _is_dangerous_rm(["--force", "/tmp/ws/file"], ws) is False

    # Recursive + force inside workspace
    assert _is_dangerous_rm(["-rf", "/tmp/ws/file"], ws) is False
    assert _is_dangerous_rm(["-rf", "file"], ws) is False
    assert _is_dangerous_rm(["--recursive", "--force", "file"], ws) is False

    # Recursive + force outside workspace
    assert _is_dangerous_rm(["-rf", "/etc/passwd"], ws) is True
    assert _is_dangerous_rm(["-rf", "../../etc/passwd"], ws) is True
    assert _is_dangerous_rm(["-rf", "$HOME/file"], ws) is True
    assert _is_dangerous_rm(["-rf", "`rm -rf /`"], ws) is True

    # With -- separator
    assert _is_dangerous_rm(["-rf", "--", "/etc/passwd"], ws) is True
    assert _is_dangerous_rm(["-rf", "--", "file"], ws) is False


def test_is_dangerous_chmod_edge_cases():
    """Test _is_dangerous_chmod with various edge cases."""
    from pathlib import Path

    from furrow.agents.worker import _is_dangerous_chmod

    ws = Path("/tmp/ws").resolve()

    # Not recursive
    assert _is_dangerous_chmod(["777", "/tmp/ws/file"], ws) is False
    assert _is_dangerous_chmod(
        ["-R", "/tmp/ws/file"], ws
    ) is True  # recursive but no force needed for chmod

    # Recursive inside workspace
    assert _is_dangerous_chmod(["-R", "777", "/tmp/ws/file"], ws) is False
    assert _is_dangerous_chmod(["--recursive", "777", "file"], ws) is False

    # Recursive outside workspace
    assert _is_dangerous_chmod(["-R", "777", "/etc/passwd"], ws) is True
    assert _is_dangerous_chmod(["-R", "777", "../../etc/passwd"], ws) is True
    assert _is_dangerous_chmod(["-R", "777", "$HOME/file"], ws) is True
    assert _is_dangerous_chmod(["-R", "777", "`echo hi`"], ws) is True

    # With -- separator
    assert _is_dangerous_chmod(["-R", "777", "--", "/etc/passwd"], ws) is True
    assert _is_dangerous_chmod(["-R", "777", "--", "file"], ws) is False


def test_extract_command_substitutions_edge_cases():
    """Test _extract_command_substitutions with various edge cases."""
    from furrow.agents.worker import _extract_command_substitutions

    # No substitutions
    assert _extract_command_substitutions("echo hello") == []
    assert _extract_command_substitutions("") == []

    # $() substitutions
    assert _extract_command_substitutions("echo $(date)") == ["date"]
    assert _extract_command_substitutions("echo $(echo $(date))") == ["echo $(date)", "date"]
    assert _extract_command_substitutions("echo $(foo) $(bar)") == ["foo", "bar"]

    # Backtick substitutions
    assert _extract_command_substitutions("echo `date`") == ["date"]
    assert _extract_command_substitutions("echo `echo `date``") == ["echo `date`", "date"]

    # Mixed
    assert _extract_command_substitutions("echo $(foo) `bar`") == ["foo", "bar"]

    # Unbalanced
    assert _extract_command_substitutions("echo $(foo") == []
    assert _extract_command_substitutions("echo `foo") == []

    # Empty substitutions
    assert _extract_command_substitutions("echo $()") == []
    assert _extract_command_substitutions("echo ``") == []


def test_expand_home_edge_cases():
    """Test _expand_home with various edge cases."""
    from furrow.agents.worker import _expand_home

    # These depend on the actual home directory, so just test they don't crash
    result = _expand_home("~/file")
    assert isinstance(result, str)
    assert len(result) > 0

    result = _expand_home("$HOME/file")
    assert isinstance(result, str)

    result = _expand_home("${HOME}/file")
    assert isinstance(result, str)

    result = _expand_home("/absolute/path")
    assert result == "/absolute/path"

    result = _expand_home("relative/path")
    assert result == "relative/path"


def test_target_inside_workspace_edge_cases():
    """Test _target_inside_workspace with various edge cases."""
    from pathlib import Path

    from furrow.agents.worker import _target_inside_workspace

    ws = Path("/tmp/ws").resolve()

    # Inside workspace
    assert _target_inside_workspace("file.txt", ws) is True
    assert _target_inside_workspace("subdir/file.txt", ws) is True
    assert _target_inside_workspace("/tmp/ws/file.txt", ws) is True
    assert _target_inside_workspace("./file.txt", ws) is True

    # Outside workspace
    assert _target_inside_workspace("/etc/passwd", ws) is False
    assert _target_inside_workspace("../../etc/passwd", ws) is False
    assert _target_inside_workspace("~/file", ws) is False  # home is outside
    assert _target_inside_workspace("$HOME/file", ws) is False

    # Special targets
    assert _target_inside_workspace("/dev/null", ws) is False  # not inside workspace
    assert _target_inside_workspace("/dev/stdout", ws) is False
    assert _target_inside_workspace("/dev/stderr", ws) is False


def test_redirects_outside_workspace_edge_cases():
    """Test _redirects_outside_workspace with various edge cases."""
    from pathlib import Path

    from furrow.agents.worker import _redirects_outside_workspace

    ws = Path("/tmp/ws").resolve()

    # Inside workspace
    assert _redirects_outside_workspace("echo hi > file.txt", ws) is False
    assert _redirects_outside_workspace("echo hi >> file.txt", ws) is False
    assert _redirects_outside_workspace("echo hi >| file.txt", ws) is False
    assert _redirects_outside_workspace("echo hi > subdir/file.txt", ws) is False

    # Outside workspace
    assert _redirects_outside_workspace("echo hi > /etc/passwd", ws) is True
    assert _redirects_outside_workspace("echo hi > ../../etc/passwd", ws) is True
    assert _redirects_outside_workspace("echo hi > ~/file", ws) is True
    assert _redirects_outside_workspace("echo hi > $HOME/file", ws) is True
    assert _redirects_outside_workspace("echo hi > `echo /etc/passwd`", ws) is True

    # Special targets (allowed)
    assert _redirects_outside_workspace("echo hi > /dev/null", ws) is False
    assert _redirects_outside_workspace("echo hi > /dev/stdout", ws) is False
    assert _redirects_outside_workspace("echo hi > /dev/stderr", ws) is False

    # Multiple redirections
    assert _redirects_outside_workspace("echo hi > file1.txt > file2.txt", ws) is False
    assert _redirects_outside_workspace("echo hi > file.txt > /etc/passwd", ws) is True


def test_writes_outside_workspace_edge_cases():
    """Test _writes_outside_workspace with various edge cases."""
    from pathlib import Path

    from furrow.agents.worker import _writes_outside_workspace

    ws = Path("/tmp/ws").resolve()

    # Not a write-path program
    assert _writes_outside_workspace("echo hi", ws) is False
    assert _writes_outside_workspace("cat file.txt", ws) is False

    # tee with path inside workspace
    assert _writes_outside_workspace("tee file.txt", ws) is False
    assert _writes_outside_workspace("tee -a file.txt", ws) is False
    assert _writes_outside_workspace("tee subdir/file.txt", ws) is False

    # tee with path outside workspace
    assert _writes_outside_workspace("tee /etc/passwd", ws) is True
    assert _writes_outside_workspace("tee ../../etc/passwd", ws) is True
    assert _writes_outside_workspace("tee ~/file", ws) is True
    assert _writes_outside_workspace("tee $HOME/file", ws) is True
    assert _writes_outside_workspace("tee `echo /etc/passwd`", ws) is True

    # Special targets (allowed)
    assert _writes_outside_workspace("tee /dev/null", ws) is False
    assert _writes_outside_workspace("tee /dev/stdout", ws) is False
    assert _writes_outside_workspace("tee /dev/stderr", ws) is False

    # Multiple path arguments (only first non-option counts)
    assert _writes_outside_workspace("tee file1.txt file2.txt", ws) is False
    assert _writes_outside_workspace("tee file.txt /etc/passwd", ws) is True


def test_is_force_push_edge_cases():
    """Test _is_force_push with various edge cases."""
    from furrow.agents.worker import _is_force_push

    # Force push variants
    assert _is_force_push("git push --force", "git push --force") is True
    assert _is_force_push("git push --force-with-lease", "git push --force-with-lease") is True
    assert _is_force_push(
        "git push --force-with-lease=origin/main",
        "git push --force-with-lease=origin/main",
    ) is True
    assert _is_force_push("git push -f origin main", "git push -f origin main") is True
    assert _is_force_push("git push origin +main", "git push origin +main") is True
    assert _is_force_push("git push origin HEAD:+main", "git push origin head:+main") is True

    # Not force push
    assert _is_force_push("git push origin main", "git push origin main") is False
    assert _is_force_push("git push -u origin main", "git push -u origin main") is False
    assert _is_force_push("ls -f", "ls -f") is False  # short -f not for git
    assert _is_force_push("git commit -f", "git commit -f") is False  # not push
    assert _is_force_push(
        "echo git push --force", "echo git push --force"
    ) is False  # not git command


def test_is_dangerous_segment_edge_cases():
    """Test _is_dangerous_segment with various edge cases."""
    from pathlib import Path

    from furrow.agents.worker import _is_dangerous_segment

    ws = Path("/tmp/ws").resolve()

    # Denied programs
    assert _is_dangerous_segment("dd if=/dev/zero", ws) is True
    assert _is_dangerous_segment("mkfs.ext4 /dev/sda", ws) is True
    assert _is_dangerous_segment("fdisk /dev/sda", ws) is True
    assert _is_dangerous_segment("parted /dev/sda", ws) is True
    assert _is_dangerous_segment("shutdown -h now", ws) is True
    assert _is_dangerous_segment("reboot", ws) is True
    assert _is_dangerous_segment("poweroff", ws) is True
    assert _is_dangerous_segment("halt", ws) is True
    assert _is_dangerous_segment("init 0", ws) is True

    # Denied patterns
    assert _is_dangerous_segment(":(){:|:&};:", ws) is True
    assert _is_dangerous_segment("rm -rf /", ws) is True
    assert _is_dangerous_segment("chmod -R 777 /", ws) is True
    assert _is_dangerous_segment("echo hi > /dev/sda", ws) is True

    # Behind wrappers
    assert _is_dangerous_segment("sudo rm -rf /", ws) is True
    assert _is_dangerous_segment("env rm -rf /", ws) is True
    assert _is_dangerous_segment("sudo chmod -R 777 /", ws) is True

    # Force push
    assert _is_dangerous_segment("git push --force", ws) is True
    assert _is_dangerous_segment("git push -f", ws) is True

    # Allowed commands
    assert _is_dangerous_segment("echo hello", ws) is False
    assert _is_dangerous_segment("ls -la", ws) is False
    assert _is_dangerous_segment("pytest tests/", ws) is False
    assert _is_dangerous_segment("git push origin main", ws) is False
    assert _is_dangerous_segment("rm file.txt", ws) is False
    assert _is_dangerous_segment("chmod 755 script.sh", ws) is False


def test_text_is_denied_edge_cases():
    """Test _text_is_denied with various edge cases."""
    from pathlib import Path

    from furrow.agents.worker import _text_is_denied

    ws = Path("/tmp/ws").resolve()

    # Direct denied patterns
    assert _text_is_denied("rm -rf /", ws, 0) is True
    assert _text_is_denied(":(){:|:&};:", ws, 0) is True

    # In substitutions
    assert _text_is_denied("echo $(rm -rf /)", ws, 0) is True
    assert _text_is_denied("echo `rm -rf /`", ws, 0) is True
    assert _text_is_denied("echo $(echo $(rm -rf /))", ws, 0) is True

    # Depth limit
    assert _text_is_denied("echo $(echo $(echo $(echo $(rm -rf /))))", ws, 0) is True  # depth 3
    assert _text_is_denied(
        "echo $(echo $(echo $(echo $(echo $(rm -rf /)))))", ws, 0
    ) is True  # depth 4 (exceeds max depth of 3)

    # Allowed
    assert _text_is_denied("echo hello", ws, 0) is False
    assert _text_is_denied("", ws, 0) is False
    assert _text_is_denied("   ", ws, 0) is False


def test_is_command_denied_edge_cases():
    """Test is_command_denied with various edge cases."""
    from pathlib import Path

    from furrow.agents.worker import is_command_denied

    ws = Path("/tmp/ws").resolve()

    # Empty
    assert is_command_denied("", ws) is False
    assert is_command_denied("   ", ws) is False
    assert is_command_denied("\n", ws) is False

    # Denied
    assert is_command_denied("rm -rf /", ws) is True
    assert is_command_denied("sudo rm -rf /", ws) is True
    assert is_command_denied("echo $(rm -rf /)", ws) is True

    # Allowed
    assert is_command_denied("echo hello", ws) is False
    assert is_command_denied("ls -la", ws) is False
    assert is_command_denied("pytest", ws) is False


def test_resolve_in_workspace_edge_cases():
    """Test resolve_in_workspace with various edge cases."""
    from pathlib import Path

    from furrow.agents.worker import resolve_in_workspace

    ws = Path("/tmp/ws").resolve()

    # Inside
    assert resolve_in_workspace(ws, "file.txt") == ws / "file.txt"
    assert resolve_in_workspace(ws, "subdir/file.txt") == ws / "subdir" / "file.txt"
    assert resolve_in_workspace(ws, "./file.txt") == ws / "file.txt"
    assert resolve_in_workspace(ws, str(ws / "file.txt")) == ws / "file.txt"

    # Outside
    assert resolve_in_workspace(ws, "../file.txt") is None
    assert resolve_in_workspace(ws, "/etc/passwd") is None
    assert resolve_in_workspace(ws, "../../etc/passwd") is None

    # Empty
    assert resolve_in_workspace(ws, "") is None
    assert resolve_in_workspace(ws, "   ") is None
