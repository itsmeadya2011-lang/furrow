from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from furrow.agents.planner import MAX_TASKS, PlannerAgent, extract_json_object
from furrow.agents.tester import TEST_COMMANDS, TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class FakeLLM(LLMClient):
    """Offline LLMClient returning pre-programmed responses."""

    def __init__(self, responses: list[str] | str, settings: Settings) -> None:
        super().__init__(settings=settings)
        if isinstance(responses, str):
            responses = [responses]
        self.responses = list(responses)
        self.calls: list[tuple[str, str | None]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append((prompt, model))
        if not self.responses:
            return ""
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


class FakeProcess:
    """Stand-in for asyncio.subprocess.Process."""

    def __init__(self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode: int | None = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = -9
        return self.returncode


class SubprocessRecorder:
    """Callable replacing asyncio.create_subprocess_exec.

    ``installed`` maps a runner tuple to the process it should return; any
    runner that is not a key raises :class:`FileNotFoundError`, exactly like a
    missing binary.
    """

    def __init__(
        self,
        installed: dict[tuple[str, ...], FakeProcess],
        missing: tuple[str, ...] = (),
    ) -> None:
        self.installed = installed
        self.missing = missing
        self.attempts: list[tuple[str, ...]] = []

    async def __call__(self, *cmd: str, **kwargs: Any) -> FakeProcess:
        key = tuple(cmd)
        self.attempts.append(key)
        if key in self.missing:
            raise FileNotFoundError(2, "No such file or directory", key[0])
        if key not in self.installed:
            raise FileNotFoundError(2, "No such file or directory", key[0])
        return self.installed[key]


def make_settings(tmp_path: Path) -> Settings:
    return Settings(workspace=tmp_path, max_parallel_tasks=5)


# --------------------------------------------------------------------------
# extract_json_object
# --------------------------------------------------------------------------


def test_extract_fenced_json() -> None:
    text = 'Here you go:\n```json\n{"summary": "ok", "files": []}\n```\nHope that helps!'
    assert extract_json_object(text) == {"summary": "ok", "files": []}


def test_extract_prose_wrapped_json() -> None:
    text = 'Sure! {"summary": "done", "files": []} -- let me know if you need more.'
    assert extract_json_object(text)["summary"] == "done"


def test_extract_bare_json() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_raises_with_raw_response() -> None:
    with pytest.raises(ValueError) as excinfo:
        extract_json_object("no json here at all")
    assert "no json here at all" in str(excinfo.value)


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_parses_fenced_json(tmp_path: Path) -> None:
    payload = {"tasks": [{"id": "1", "description": "do a thing"}], "rationale": "because"}
    client = FakeLLM(f"```json\n{json.dumps(payload)}\n```", make_settings(tmp_path))
    plan = await PlannerAgent(client=client).plan("goal")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "do a thing"
    assert plan.rationale == "because"


@pytest.mark.asyncio
async def test_planner_ignores_unknown_keys(tmp_path: Path) -> None:
    raw = {
        "tasks": [
            {
                "id": "1",
                "description": "task one",
                "unknown_key": "dropped",
                "effort": 5,
            }
        ],
        "rationale": "r",
        "confidence": 0.9,
    }
    client = FakeLLM(json.dumps(raw), make_settings(tmp_path))
    plan = await PlannerAgent(client=client).plan("goal")
    assert not hasattr(plan, "confidence")
    assert plan.tasks[0].model_dump().get("unknown_key") is None
    assert "unknown_key" not in plan.tasks[0].model_dump()


@pytest.mark.asyncio
async def test_planner_truncates_to_five_tasks(tmp_path: Path) -> None:
    raw = {
        "tasks": [
            {"id": str(i), "description": f"task {i}"} for i in range(1, 9)
        ],
        "rationale": "too many",
    }
    client = FakeLLM(json.dumps(raw), make_settings(tmp_path))
    plan = await PlannerAgent(client=client).plan("goal")
    assert len(plan.tasks) == MAX_TASKS == 5
    assert [t.id for t in plan.tasks] == ["1", "2", "3", "4", "5"]


@pytest.mark.asyncio
async def test_planner_drops_task_without_description(tmp_path: Path) -> None:
    raw = {
        "tasks": [
            {"id": "1", "description": "keep me"},
            {"id": "2"},
            {"description": "no id"},
            "not a dict",
            {"id": "3", "description": "keep me too"},
        ],
        "rationale": "r",
    }
    client = FakeLLM(json.dumps(raw), make_settings(tmp_path))
    plan = await PlannerAgent(client=client).plan("goal")
    assert [t.description for t in plan.tasks] == ["keep me", "keep me too"]


@pytest.mark.asyncio
async def test_planner_raises_on_unparseable_response(tmp_path: Path) -> None:
    client = FakeLLM("I refuse to answer.", make_settings(tmp_path))
    with pytest.raises(ValueError):
        await PlannerAgent(client=client).plan("goal")


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_writes_files_in_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    payload = {
        "summary": "added a module",
        "files": [
            {"path": "pkg/mod.py", "content": "VALUE = 1\n"},
            {"path": "top.txt", "content": "hello"},
        ],
    }
    client = FakeLLM(json.dumps(payload), make_settings(workspace))
    task = TaskModel(id="1", description="add a module")
    result = await WorkerAgent(task=task, client=client).run()

    assert (workspace / "pkg" / "mod.py").read_text() == "VALUE = 1\n"
    assert (workspace / "top.txt").read_text() == "hello"
    assert "added a module" in result
    assert "pkg/mod.py" in result
    assert client.calls, "worker must actually call the LLM"


@pytest.mark.asyncio
async def test_worker_parses_fenced_json_output(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    payload = {"summary": "done", "files": [{"path": "a/b.py", "content": "x = 2"}]}
    client = FakeLLM(f"```json\n{json.dumps(payload)}\n```", make_settings(workspace))
    await WorkerAgent(task=TaskModel(id="1", description="d"), client=client).run()
    assert (workspace / "a" / "b.py").read_text() == "x = 2"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_path", ["../escape.py", "/etc/passwd"])
async def test_worker_rejects_unsafe_paths(tmp_path: Path, bad_path: str) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    escape_target = workspace.parent / "escape.py"
    payload = {
        "summary": "trying something",
        "files": [{"path": bad_path, "content": "pwned"}],
    }
    client = FakeLLM(json.dumps(payload), make_settings(workspace))
    result = await WorkerAgent(task=TaskModel(id="1", description="d"), client=client).run()

    assert "ejected unsafe path" in result
    assert not escape_target.exists()
    assert not (workspace / "escape.py").exists()
    if bad_path == "/etc/passwd":
        assert Path("/etc/passwd").read_text().strip() != "pwned"


@pytest.mark.asyncio
async def test_worker_rejects_unsafe_path_but_still_writes_safe_one(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    payload = {
        "summary": "mixed",
        "files": [
            {"path": "../escape.py", "content": "bad"},
            {"path": "safe.py", "content": "good"},
        ],
    }
    client = FakeLLM(json.dumps(payload), make_settings(workspace))
    result = await WorkerAgent(task=TaskModel(id="1", description="d"), client=client).run()

    assert (workspace / "safe.py").read_text() == "good"
    assert not (workspace.parent / "escape.py").exists()
    assert "safe.py" in result
    assert "escape.py" in result


@pytest.mark.asyncio
async def test_worker_returns_raw_response_when_unparseable(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    client = FakeLLM("I could not use JSON, sorry.", make_settings(workspace))
    result = await WorkerAgent(task=TaskModel(id="1", description="d"), client=client).run()
    assert result == "I could not use JSON, sorry."
    assert list(workspace.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_returns_summary_when_no_files(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    payload = {"summary": "nothing to change", "files": []}
    client = FakeLLM(json.dumps(payload), make_settings(workspace))
    result = await WorkerAgent(task=TaskModel(id="1", description="d"), client=client).run()
    assert result == "nothing to change"
    assert list(workspace.iterdir()) == []


# --------------------------------------------------------------------------
# Tester
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tester_exit_code_beats_optimistic_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = SubprocessRecorder(
        {TEST_COMMANDS[0]: FakeProcess(returncode=1, stdout=b"", stderr=b"2 failed")}
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", recorder)

    llm_reply = json.dumps({"passed": True, "summary": "All good!", "failures": []})
    client = FakeLLM(llm_reply, make_settings(tmp_path))
    result = await TesterAgent(client=client).run("goal", [TaskModel(id="1", description="d")])

    assert result.passed is False, "nonzero exit code must win over the model's claim"
    assert list(recorder.attempts) == [TEST_COMMANDS[0]]


@pytest.mark.asyncio
async def test_tester_passes_on_zero_exit_with_unhelpful_llm_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = SubprocessRecorder({TEST_COMMANDS[0]: FakeProcess(returncode=0, stdout=b"1 passed")})
    monkeypatch.setattr(asyncio, "create_subprocess_exec", recorder)

    client = FakeLLM("Looks fine to me, good job!", make_settings(tmp_path))
    result = await TesterAgent(client=client).run("goal", [TaskModel(id="1", description="d")])

    assert result.passed is True


@pytest.mark.asyncio
async def test_tester_falls_through_when_runner_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert len(TEST_COMMANDS) >= 2
    good = FakeProcess(returncode=0, stdout=b"42 passed")
    recorder = SubprocessRecorder({TEST_COMMANDS[1]: good})
    monkeypatch.setattr(asyncio, "create_subprocess_exec", recorder)

    llm_reply = json.dumps({"passed": True, "summary": "ok", "failures": []})
    client = FakeLLM(llm_reply, make_settings(tmp_path))
    result = await TesterAgent(client=client).run("goal", [TaskModel(id="1", description="d")])

    assert list(recorder.attempts) == [TEST_COMMANDS[0], TEST_COMMANDS[1]]
    assert result.passed is True
    assert "42 passed" in client.calls[0][0], "runner output must reach the model"


@pytest.mark.asyncio
async def test_tester_reports_failure_when_no_runner_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = SubprocessRecorder({})
    monkeypatch.setattr(asyncio, "create_subprocess_exec", recorder)

    llm_reply = json.dumps({"passed": False, "summary": "nothing ran", "failures": []})
    client = FakeLLM(llm_reply, make_settings(tmp_path))
    result = await TesterAgent(client=client).run("goal", [TaskModel(id="1", description="d")])

    assert len(recorder.attempts) == len(TEST_COMMANDS)
    assert result.passed is False
    assert "No test runner found." in client.calls[0][0]