"""Tests for :mod:`furrow.agents` (planner, tester, worker).

Everything here is hermetic: no network, no real LLM calls, no API keys and no
subprocess is ever spawned.  ``LLMClient`` is replaced by :class:`FakeClient`, and
the tester's subprocess plumbing is replaced by tiny stand-ins for
``shutil.which`` / ``asyncio.create_subprocess_exec``.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TEST_RUNNER_CANDIDATES, TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel

# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class FakeClient:
    """Stand-in for :class:`furrow.llm.LLMClient` that records what it was asked."""

    def __init__(self, settings: Settings | None = None, response: str = "") -> None:
        self.settings = settings if settings is not None else Settings()
        self.response = response
        self.prompts: list[str] = []
        self.models: list[str | None] = []
        self.systems: list[str] = []
        self.writes: list[tuple[Path, str]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        self.models.append(model)
        return self.response

    async def write_file(self, path: str | Path, content: str) -> None:
        self.writes.append((Path(path), content))


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace=tmp_path,
        planner_model="planner-model-x",
        worker_model="worker-model-y",
        tester_model="tester-model-z",
    )


def _fenced(payload: str) -> str:
    return "Working on it.\n\n```json\n" + payload + "\n```\n"


# --------------------------------------------------------------------------------------
# PlannerAgent
# --------------------------------------------------------------------------------------

PLAN_JSON = """{
  "tasks": [
    {
      "id": "1",
      "description": "Add the auth module",
      "files": ["src/auth.py"],
      "dependencies": []
    },
    {
      "id": "2",
      "description": "Add auth tests",
      "files": ["tests/test_auth.py"],
      "dependencies": ["1"]
    }
  ],
  "rationale": "Two small independent-ish slices."
}"""


async def test_planner_parses_fenced_json_into_plan(tmp_path) -> None:
    client = FakeClient(_settings(tmp_path), response=_fenced(PLAN_JSON))
    agent = PlannerAgent(client)

    plan = await agent.plan("Add authentication")

    assert [t.id for t in plan.tasks] == ["1", "2"]
    assert plan.rationale == "Two small independent-ish slices."
    assert plan.tasks[0].files == ["src/auth.py"]
    assert plan.tasks[1].dependencies == ["1"]
    # Defaults still apply to fields the LLM omitted.
    assert plan.tasks[0].status == "pending"
    assert plan.tasks[0].result is None


async def test_planner_includes_goal_and_uses_planner_model(tmp_path) -> None:
    client = FakeClient(_settings(tmp_path), response=_fenced(PLAN_JSON))
    agent = PlannerAgent(client)

    await agent.plan("Add authentication")

    assert len(client.prompts) == 1
    assert "Add authentication" in client.prompts[0]
    assert client.models == ["planner-model-x"]


async def test_planner_appends_context_when_supplied(tmp_path) -> None:
    client = FakeClient(_settings(tmp_path), response=_fenced(PLAN_JSON))
    agent = PlannerAgent(client)

    await agent.plan("Add authentication", context="  previous failure: missing import  ")

    prompt = client.prompts[0]
    assert "previous failure: missing import" in prompt
    assert "Previous cycle failures to address:" in prompt
    # Stripped, not double-spaced.
    assert "  previous failure" not in prompt


@pytest.mark.parametrize("context", [None, "", "   \n  "])
async def test_planner_omits_blank_context(tmp_path, context: str | None) -> None:
    client = FakeClient(_settings(tmp_path), response=_fenced(PLAN_JSON))
    agent = PlannerAgent(client)

    await agent.plan("Add authentication", context=context)

    assert "Previous cycle failures to address:" not in client.prompts[0]


async def test_planner_unparseable_response_raises_value_error(tmp_path) -> None:
    client = FakeClient(_settings(tmp_path), response="I am afraid I cannot do that.")
    agent = PlannerAgent(client)

    with pytest.raises(ValueError, match="Failed to parse plan"):
        await agent.plan("Add authentication")


async def test_planner_missing_rationale_raises_validation_error(tmp_path) -> None:
    """pydantic's ``ValidationError`` is a ``ValueError`` subclass, so the planner wraps it."""
    assert issubclass(ValidationError, ValueError)

    with pytest.raises(ValidationError) as direct:
        from furrow.config import Plan

        Plan(tasks=[])  # type: ignore[call-arg]
    assert "rationale" in str(direct.value)

    client = FakeClient(_settings(tmp_path), response='{"tasks": []}')
    agent = PlannerAgent(client)

    with pytest.raises(ValueError) as wrapped:
        await agent.plan("Add authentication")
    assert "Failed to parse plan" in str(wrapped.value)
    assert isinstance(wrapped.value, ValueError)


# --------------------------------------------------------------------------------------
# TesterAgent._run_tests
# --------------------------------------------------------------------------------------


class FakeProc:
    """Minimal stand-in for ``asyncio.subprocess.Process``."""

    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.killed = False
        self.waited = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        self.waited = True
        return self.returncode


def _spawner(scripted: dict[tuple[str, ...], object]) -> tuple[object, list[tuple[str, ...]]]:
    """Return ``(create_subprocess_exec, spawned_argv_log)`` for the given script.

    Values in ``scripted`` are either a :class:`FakeProc` to return or an exception
    instance to raise.
    """
    spawned: list[tuple[str, ...]] = []

    async def create(*cmd: str, **kwargs: object) -> FakeProc:
        spawned.append(cmd)
        outcome = scripted[tuple(cmd)]
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, FakeProc)
        return outcome

    return create, spawned


def _which(*available: str):
    def fake_which(name: str, *args: object, **kwargs: object) -> str | None:
        return f"/usr/bin/{name}" if name in available else None

    return fake_which


def _install_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: tuple[str, ...],
    scripted: dict[tuple[str, ...], object],
) -> list[tuple[str, ...]]:
    monkeypatch.setattr(shutil, "which", _which(*available))
    create, spawned = _spawner(scripted)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    return spawned


async def test_run_tests_falls_through_to_first_success(tmp_path, monkeypatch) -> None:
    pytest_cmd = tuple(TEST_RUNNER_CANDIDATES[0])
    python_cmd = tuple(TEST_RUNNER_CANDIDATES[1])
    assert python_cmd == ("python", "-m", "pytest", "-q")

    spawned = _install_runner(
        monkeypatch,
        available=("pytest", "python"),
        scripted={
            pytest_cmd: FakeProc(1, stdout=b"first attempt output\n"),
            python_cmd: FakeProc(0, stdout=b"second attempt output\n"),
        },
    )

    out = await TesterAgent(FakeClient(_settings(tmp_path)))._run_tests()

    assert out == "$ python -m pytest -q exit=0\nsecond attempt output\n"
    assert spawned == [pytest_cmd, python_cmd]


async def test_run_tests_returns_last_failure_when_nothing_succeeds(tmp_path, monkeypatch) -> None:
    python_cmd = tuple(TEST_RUNNER_CANDIDATES[1])
    _install_runner(
        monkeypatch,
        available=("pytest", "python"),
        scripted={
            tuple(TEST_RUNNER_CANDIDATES[0]): FakeProc(1, stdout=b"boom\n"),
            python_cmd: FakeProc(2, stdout=b"boom2\n", stderr=b"traceback here\n"),
        },
    )

    out = await TesterAgent(FakeClient(_settings(tmp_path)))._run_tests()

    assert out == "$ python -m pytest -q exit=2\nboom2\ntraceback here\n"


async def test_run_tests_reports_missing_runner(tmp_path, monkeypatch) -> None:
    spawned = _install_runner(monkeypatch, available=(), scripted={})

    out = await TesterAgent(FakeClient(_settings(tmp_path)))._run_tests()

    assert out == "No test runner found."
    assert spawned == []


async def test_run_tests_skips_candidate_that_cannot_spawn(tmp_path, monkeypatch) -> None:
    python_cmd = tuple(TEST_RUNNER_CANDIDATES[1])
    spawned = _install_runner(
        monkeypatch,
        available=("pytest", "python"),
        scripted={
            tuple(TEST_RUNNER_CANDIDATES[0]): OSError("Exec format error"),
            python_cmd: FakeProc(0, stdout=b"recovered\n"),
        },
    )

    out = await TesterAgent(FakeClient(_settings(tmp_path)))._run_tests()

    assert out == "$ python -m pytest -q exit=0\nrecovered\n"
    assert len(spawned) == 2


async def test_run_tests_all_candidates_failing_to_spawn_reports_missing_runner(
    tmp_path, monkeypatch
) -> None:
    script = {tuple(cmd): OSError("boom") for cmd in TEST_RUNNER_CANDIDATES}
    _install_runner(
        monkeypatch,
        available=("pytest", "python", "npm", "pnpm", "yarn", "cargo", "go"),
        scripted=script,
    )

    out = await TesterAgent(FakeClient(_settings(tmp_path)))._run_tests()

    assert out == "No test runner found."


# --------------------------------------------------------------------------------------
# TesterAgent pass/fail heuristic
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    ["Not passed", "1 failed", "error", "Traceback (most recent call last):", ""],
)
def test_looks_passed_is_false_for_failure_markers(response: str) -> None:
    assert TesterAgent._looks_passed(response) is False


@pytest.mark.parametrize(
    "response",
    ["3 passed in 0.02s", "All tests passed", "PASSED"],
)
def test_looks_passed_is_true_for_clean_output(response: str) -> None:
    assert TesterAgent._looks_passed(response) is True


# --------------------------------------------------------------------------------------
# TesterAgent.run
# --------------------------------------------------------------------------------------


async def test_tester_run_returns_parsed_json_result(tmp_path, monkeypatch) -> None:
    _install_runner(monkeypatch, available=(), scripted={})
    client = FakeClient(
        _settings(tmp_path),
        response='```json\n{"passed": false, "summary": "one test failed", '
        '"failures": ["tests/test_a.py::test_x"]}\n```',
    )

    result = await TesterAgent(client).run("goal", [])

    assert result.passed is False
    assert result.summary == "one test failed"
    assert result.failures == ["tests/test_a.py::test_x"]
    assert "No test runner found." in client.prompts[0]
    assert client.models == ["tester-model-z"]


async def test_tester_run_falls_back_to_heuristic_without_json(tmp_path, monkeypatch) -> None:
    _install_runner(monkeypatch, available=(), scripted={})
    client = FakeClient(_settings(tmp_path), response="3 passed in 0.01s")

    result = await TesterAgent(client).run("goal", [])

    assert result.passed is True
    assert result.summary == "3 passed in 0.01s"
    assert result.failures == []


# --------------------------------------------------------------------------------------
# WorkerAgent
# --------------------------------------------------------------------------------------

EDIT_PAYLOAD = (
    '{"summary": "Added the app entrypoint.", '
    '"edits": [{"path": "src/app.py", "content": "print(1)\\n"}]}'
)


async def test_worker_writes_edits_and_reports_summary(tmp_path) -> None:
    client = FakeClient(_settings(tmp_path), response=_fenced(EDIT_PAYLOAD))
    agent = WorkerAgent(TaskModel(id="1", description="add entrypoint"), client)

    result = await agent.run()

    expected = (tmp_path / "src" / "app.py").resolve()
    assert [(p, c) for p, c in client.writes] == [(expected, "print(1)\n")]
    assert "Added the app entrypoint." in result
    assert "Files written: src/app.py" in result
    assert "add entrypoint" in client.prompts[0]
    assert client.models == ["worker-model-y"]


@pytest.mark.parametrize("escape", ["../../etc/passwd", "/etc/passwd"])
async def test_worker_rejects_edits_outside_workspace(tmp_path, escape: str) -> None:
    payload = (
        '{"summary": "Trying to escape.", '
        f'"edits": [{{"path": "{escape}", "content": "pwned"}}]}}'
    )
    client = FakeClient(_settings(tmp_path), response=_fenced(payload))
    agent = WorkerAgent(TaskModel(id="1", description="escape attempt"), client)

    result = await agent.run()

    assert client.writes == []
    assert "Rejected edits outside workspace" in result
    assert escape in result
    assert "Files written:" not in result


async def test_worker_without_json_block_returns_raw_prose(tmp_path) -> None:
    prose = "I could not make any changes because the task was already done."
    client = FakeClient(_settings(tmp_path), response=prose)
    agent = WorkerAgent(TaskModel(id="1", description="noop"), client)

    result = await agent.run()

    assert result == prose
    assert client.writes == []


async def test_worker_with_missing_edits_key_reports_summary_only(tmp_path) -> None:
    client = FakeClient(_settings(tmp_path), response=_fenced('{"summary": "Nothing to do."}'))
    agent = WorkerAgent(TaskModel(id="1", description="noop"), client)

    result = await agent.run()

    assert result == "Nothing to do."
    assert client.writes == []


async def test_worker_with_non_dict_payload_returns_raw_prose(tmp_path) -> None:
    response = _fenced("[1, 2, 3]")
    client = FakeClient(_settings(tmp_path), response=response)
    agent = WorkerAgent(TaskModel(id="1", description="odd"), client)

    assert await agent.run() == response
    assert client.writes == []


async def test_worker_reports_malformed_edit_without_writing(tmp_path) -> None:
    payload = (
        '{"summary": "Bad edits.", "edits": ['
        '{"path": "", "content": "x"}, '
        '{"path": "src/ok.py", "content": 42}, '
        '"not-a-dict"]}'
    )
    client = FakeClient(_settings(tmp_path), response=_fenced(payload))
    agent = WorkerAgent(TaskModel(id="1", description="malformed"), client)

    result = await agent.run()

    assert client.writes == []
    assert "Rejected edits outside workspace" in result
    assert "<missing path>" in result


async def test_worker_mixed_inside_and_outside_edits(tmp_path) -> None:
    payload = (
        '{"summary": "Mixed.", "edits": ['
        '{"path": "src/inside.py", "content": "ok"}, '
        '{"path": "../outside.py", "content": "no"}]}'
    )
    client = FakeClient(_settings(tmp_path), response=_fenced(payload))
    agent = WorkerAgent(TaskModel(id="1", description="mixed"), client)

    result = await agent.run()

    assert [str(p) for p, _ in client.writes] == [str((tmp_path / "src" / "inside.py").resolve())]
    assert "Files written: src/inside.py" in result
    assert "Rejected edits outside workspace: ../outside.py" in result
    assert (tmp_path.parent / "outside.py").exists() is False


def test_worker_inside_helper_allows_workspace_itself(tmp_path) -> None:
    workspace = tmp_path.resolve()
    assert WorkerAgent._inside(workspace, workspace) is True
    assert WorkerAgent._inside(workspace / "a" / "b.py", workspace) is True
    assert WorkerAgent._inside(workspace.parent / "b.py", workspace) is False
    # A sibling directory sharing a name prefix is still outside.
    assert WorkerAgent._inside(Path(f"{workspace}_evil/b.py"), workspace) is False