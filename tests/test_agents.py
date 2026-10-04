"""Offline unit tests for the agent layer: JSON recovery, prompts, and real checks.

Everything here is fake, so the suite stays fast and hermetic:

* every agent gets a duck-typed client whose ``complete`` replays a canned string,
  so no provider SDK client is ever constructed and no network call is made;
* every subprocess goes through a fake ``asyncio.create_subprocess_exec`` that
  raises ``FileNotFoundError`` for unregistered executables, so ``ruff``, ``npm``,
  ``cargo`` and ``go`` are never really spawned and the pytest suite is never
  really run against this repository;
* the per-check wall-clock bound is monkeypatched down to ``FAKE_TIMEOUT`` for
  the timeout test instead of waiting out the real 60 seconds.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from furrow.agents import prompts
from furrow.agents.json_parse import MAX_EXCERPT_CHARS, parse_json_object, strip_code_fences
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import (
    NO_CHECKS_MESSAGE,
    TIMEOUT_RETURNCODE,
    CheckResult,
    TesterAgent,
    format_check_results,
)
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient

GOAL = "Add a health endpoint to the service"

#: Sub-second stand-in for the real 60s per-check budget.
FAKE_TIMEOUT = 0.01

PYTHON_TOOLS = ("ruff", "mypy")


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeProcess:
    """Minimal stand-in for ``asyncio.subprocess.Process``."""

    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        hang: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode: int | None = returncode
        self.hang = hang
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.hang:
            # Far longer than FAKE_TIMEOUT, so the harness budget always wins and
            # the coroutine is cancelled instead of actually sleeping.
            await asyncio.sleep(30)
        return self.stdout, self.stderr

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode if self.returncode is not None else 0


class SpawnRecorder:
    """Fake ``asyncio.create_subprocess_exec`` with per-executable outcomes.

    An executable that was never registered raises ``FileNotFoundError``, exactly
    like the OS does for a tool that is not installed.  Every attempt is recorded
    so a test can assert what was (and was not) spawned.
    """

    def __init__(self, registered: dict[str, FakeProcess] | None = None) -> None:
        self.registered: dict[str, FakeProcess] = dict(registered or {})
        self.calls: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []

    def add(
        self,
        executable: str,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        hang: bool = False,
    ) -> FakeProcess:
        process = FakeProcess(
            stdout=stdout, stderr=stderr, returncode=returncode, hang=hang
        )
        self.registered[executable] = process
        return process

    def add_python_toolchain(self, *, returncode: int = 0, stdout: bytes = b"") -> None:
        for tool in PYTHON_TOOLS:
            self.add(tool, returncode=returncode, stdout=stdout)

    async def __call__(self, *argv: str, **kwargs: Any) -> FakeProcess:
        self.calls.append(list(argv))
        self.kwargs.append(kwargs)
        executable = argv[0] if argv else ""
        if executable not in self.registered:
            raise FileNotFoundError(2, "No such file or directory", executable)
        return self.registered[executable]

    @property
    def executables(self) -> list[str]:
        return [call[0] for call in self.calls if call]


class FakeClient:
    """Duck-typed ``LLMClient`` stand-in: records prompts, replays canned replies."""

    def __init__(self, settings: Settings, replies: list[str] | None = None) -> None:
        self.settings = settings
        self.replies = list(replies or [])
        self.prompts: list[str] = []
        self.models: list[str | None] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        self.models.append(model)
        if not self.replies:
            return ""
        index = min(len(self.prompts) - 1, len(self.replies) - 1)
        return self.replies[index]

    @property
    def last_prompt(self) -> str:
        assert self.prompts, "the agent never called the LLM"
        return self.prompts[-1]


def _settings(workspace: Path | None = None, **overrides: Any) -> Settings:
    values: dict[str, Any] = {"max_cycles": 1}
    if workspace is not None:
        values["workspace"] = workspace
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _task(
    task_id: str = "1",
    description: str = "do the work",
    files: list[str] | None = None,
    dependencies: list[str] | None = None,
) -> TaskModel:
    return TaskModel(
        id=task_id,
        description=description,
        files=list(files or []),
        dependencies=list(dependencies or []),
    )


def _marker(workspace: Path, name: str) -> Path:
    (workspace / name).write_text(f"{name} marker\n", encoding="utf-8")
    return workspace


@pytest.fixture
def python_workspace(tmp_path: Path) -> Path:
    """A workspace holding only ``pyproject.toml`` → pytest, ruff, mypy apply."""
    return _marker(tmp_path, "pyproject.toml")


def _install_spawns(monkeypatch: pytest.MonkeyPatch, spawner: SpawnRecorder) -> SpawnRecorder:
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawner)
    return spawner


def _agent_for(workspace: Path) -> TesterAgent:
    return TesterAgent(client=FakeClient(_settings(workspace)))  # type: ignore[arg-type]


def _planned_names(workspace: Path, marker: str | None = None) -> list[str]:
    if marker is not None:
        # One fresh workspace per marker so they never accumulate.
        workspace = workspace / marker.replace(".", "-")
        workspace.mkdir()
        _marker(workspace, marker)
    return [name for name, _command in _agent_for(workspace)._plan_checks()]


def _planned_commands(workspace: Path) -> dict[str, list[str]]:
    return dict(_agent_for(workspace)._plan_checks())


def _all_green_spawns() -> SpawnRecorder:
    """A spawner where pytest/ruff/mypy all exist and exit zero."""
    spawner = SpawnRecorder()
    spawner.add(sys.executable, returncode=0, stdout=b"3 passed in 0.02s\n")
    spawner.add_python_toolchain(returncode=0)
    return spawner


# --------------------------------------------------------------------------
# 1. parse_json_object / strip_code_fences
# --------------------------------------------------------------------------


def test_parses_a_plain_json_object() -> None:
    assert parse_json_object('{"tasks": [], "rationale": "nothing to do"}') == {
        "tasks": [],
        "rationale": "nothing to do",
    }


def test_parses_a_fenced_json_object() -> None:
    reply = '```json\n{"passed": true, "summary": "all green", "failures": []}\n```'
    assert parse_json_object(reply) == {
        "passed": True,
        "summary": "all green",
        "failures": [],
    }


def test_parses_json_padded_with_prose_on_both_sides() -> None:
    reply = (
        "Sure, here is the plan you asked for:\n\n"
        '```json\n{"tasks": [], "rationale": "already implemented"}\n```\n\n'
        "Let me know if you would like it broken down further."
    )
    assert parse_json_object(reply) == {"tasks": [], "rationale": "already implemented"}


def test_prose_wrapped_json_without_a_fence_still_parses() -> None:
    reply = 'Thinking about it, I would do {"passed": false, "summary": "red", "failures": []}.'
    assert parse_json_object(reply)["passed"] is False


def test_braces_inside_strings_do_not_truncate_the_object() -> None:
    reply = (
        "Result: "
        '{"summary": "wrap the value in {braces}", "nested": {"a": {"b": 1}}, "n": 3}'
        " -- hope that helps."
    )
    assert parse_json_object(reply) == {
        "summary": "wrap the value in {braces}",
        "nested": {"a": {"b": 1}},
        "n": 3,
    }


def test_escaped_quotes_inside_strings_do_not_confuse_the_scanner() -> None:
    reply = (
        'Here you go:\n{"summary": "say \\"hi\\" then } stop", "failures": ["a } brace"]}\nthanks'
    )
    assert parse_json_object(reply) == {
        "summary": 'say "hi" then } stop',
        "failures": ["a } brace"],
    }


def test_non_json_reply_raises_a_descriptive_value_error() -> None:
    reply = "I could not complete this task, sorry." * 20
    with pytest.raises(ValueError) as excinfo:
        parse_json_object(reply)
    message = str(excinfo.value)
    assert "Could not parse a JSON object" in message
    assert "I could not complete this task" in message


def test_json_array_reply_is_rejected_with_the_reason() -> None:
    with pytest.raises(ValueError, match="top-level value is list"):
        parse_json_object("[1, 2, 3]")


def test_error_excerpt_is_whitespace_collapsed_and_length_bounded() -> None:
    reply = "nothing structured here\n" + ("blah blah " * 200)
    with pytest.raises(ValueError) as excinfo:
        parse_json_object(reply)
    excerpt = str(excinfo.value).split("Reply excerpt: ", 1)[1]
    assert "\n" not in excerpt
    assert excerpt.endswith("...'")
    # repr() adds the two quotes plus the three-dot truncation marker.
    assert len(excerpt) <= MAX_EXCERPT_CHARS + 6


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ("```\nplain body\n```", "plain body"),
        ("```python\nprint(1)\n```", "print(1)"),
        ('  {"a": 1}  ', '{"a": 1}'),
        ("no fence at all", "no fence at all"),
    ],
)
def test_strip_code_fences(reply: str, expected: str) -> None:
    assert strip_code_fences(reply) == expected


# --------------------------------------------------------------------------
# 2. PlannerAgent
# --------------------------------------------------------------------------

PLANNER_PAYLOAD = {
    "tasks": [
        {
            "id": "1",
            "description": "Add the health endpoint",
            "files": ["furrow/web/server.py"],
            "dependencies": [],
        },
        {
            "id": "2",
            "description": "Add the endpoint test",
            "files": ["tests/test_entrypoints.py"],
            "dependencies": ["1"],
        },
    ],
    "rationale": "endpoint first, then its test",
}

PLANNER_FENCED_REPLY = (
    "Here is the plan:\n\n```json\n" + json.dumps(PLANNER_PAYLOAD) + "\n```\n\nThat is the plan."
)


async def test_plan_parses_a_fenced_reply_with_prose() -> None:
    settings = _settings()
    client = FakeClient(settings, [PLANNER_FENCED_REPLY])

    plan = await PlannerAgent(client=client).plan(GOAL)  # type: ignore[arg-type]

    assert isinstance(plan, Plan)
    assert plan.rationale == "endpoint first, then its test"
    assert [task.id for task in plan.tasks] == ["1", "2"]
    assert plan.tasks[0].description == "Add the health endpoint"
    assert plan.tasks[0].files == ["furrow/web/server.py"]
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.tasks[0].status == "pending"
    assert plan.tasks[1].result is None
    # The goal and the planner-specific model both reached the LLM.
    assert GOAL in client.last_prompt
    assert client.models == [settings.planner_model]


async def test_plan_with_an_empty_task_list_parses_and_signals_a_halt() -> None:
    reply = '```json\n{"tasks": [], "rationale": "the goal is already satisfied"}\n```'
    client = FakeClient(_settings(), [reply])

    plan = await PlannerAgent(client=client).plan(GOAL)  # type: ignore[arg-type]

    assert plan.tasks == []
    assert plan.rationale == "the goal is already satisfied"


async def test_unparseable_plan_reply_raises_naming_the_response() -> None:
    reply = "I'm afraid I could not produce a plan."
    client = FakeClient(_settings(), [reply])

    with pytest.raises(ValueError) as excinfo:
        await PlannerAgent(client=client).plan(GOAL)  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "Failed to parse plan from LLM" in message
    assert reply in message


async def test_plan_missing_required_keys_raises_naming_the_response() -> None:
    reply = '{"tasks": []}'
    client = FakeClient(_settings(), [reply])

    with pytest.raises(ValueError) as excinfo:
        await PlannerAgent(client=client).plan(GOAL)  # type: ignore[arg-type]

    assert "Failed to parse plan from LLM" in str(excinfo.value)
    assert reply in str(excinfo.value)


# --------------------------------------------------------------------------
# 3. Construction (client=None regression) and prompt contracts
# --------------------------------------------------------------------------


def test_agents_construct_with_client_and_settings_none() -> None:
    # Regression: these forwarded ``settings=None`` into ``LLMClient``, which
    # raised AttributeError while binding its non-optional default argument.
    planner = PlannerAgent(client=None, settings=None)
    worker = WorkerAgent(_task(), client=None, settings=None)
    tester = TesterAgent(client=None, settings=None)

    for agent in (planner, worker, tester):
        assert isinstance(agent.client, LLMClient)
        assert isinstance(agent.client.settings, Settings)
    assert isinstance(tester.workspace, Path)


def test_agents_construct_with_an_explicit_settings_object() -> None:
    settings = _settings()
    agents = [
        PlannerAgent(client=None, settings=settings),
        WorkerAgent(_task(), client=None, settings=settings),
        TesterAgent(client=None, settings=settings),
    ]

    for agent in agents:
        assert agent.client.settings is settings
        # No provider SDK client exists yet, so nothing reached the network.
        assert agent.client._anthropic is None
        assert agent.client._openai is None


def test_a_supplied_client_is_used_as_is() -> None:
    settings = _settings()
    client = FakeClient(settings)

    assert PlannerAgent(client=client).client is client  # type: ignore[arg-type]
    assert WorkerAgent(_task(), client=client).client is client  # type: ignore[arg-type]
    assert TesterAgent(client=client).client is client  # type: ignore[arg-type]


def test_planner_prompt_demands_the_exact_json_shape() -> None:
    assert "JSON only" in prompts.PLANNER_PROMPT
    assert '"tasks"' in prompts.PLANNER_PROMPT
    assert '"rationale"' in prompts.PLANNER_PROMPT
    assert '"dependencies"' in prompts.PLANNER_PROMPT
    assert '"files"' in prompts.PLANNER_PROMPT
    # The halt signal (an empty task list) must be documented for the model.
    assert "empty" in prompts.PLANNER_PROMPT


def test_tester_prompt_makes_exit_codes_authoritative() -> None:
    assert "authoritative" in prompts.TESTER_PROMPT
    assert '"failures"' in prompts.TESTER_PROMPT
    assert "UNVERIFIED" in prompts.TESTER_PROMPT
    assert "Never report" in prompts.TESTER_PROMPT


def test_worker_prompt_forbids_subagents_and_unverified_claims() -> None:
    assert "Do not spawn subagents." in prompts.WORKER_PROMPT
    assert "minimal, targeted changes" in prompts.WORKER_PROMPT
    assert "Never claim a check passed" in prompts.WORKER_PROMPT


# --------------------------------------------------------------------------
# 4. WorkerAgent
# --------------------------------------------------------------------------


async def test_worker_prompt_carries_the_task_and_its_files() -> None:
    settings = _settings()
    client = FakeClient(settings, ["added the route"])
    task = _task(
        task_id="7",
        description="Wire the health route",
        files=["furrow/web/server.py", "furrow/cli/main.py"],
    )

    result = await WorkerAgent(task=task, client=client).run()  # type: ignore[arg-type]

    assert result == "added the route"
    prompt = client.last_prompt
    assert "Wire the health route" in prompt
    assert "furrow/web/server.py" in prompt
    assert "furrow/cli/main.py" in prompt
    assert prompts.WORKER_PROMPT in prompt
    assert client.models == [settings.worker_model]


async def test_worker_prompt_says_any_when_the_task_names_no_files() -> None:
    client = FakeClient(_settings(), ["nothing to change"])

    await WorkerAgent(task=_task(), client=client).run()  # type: ignore[arg-type]

    assert "Files to touch: any" in client.last_prompt


# --------------------------------------------------------------------------
# 5. Tester truthfulness
# --------------------------------------------------------------------------


async def test_nonzero_exit_code_overrides_an_llm_that_claims_success(
    python_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawner = SpawnRecorder()
    spawner.add(
        sys.executable,
        returncode=1,
        stdout=b"",
        stderr=b"tests/test_x.py::test_y FAILED\n2 failed, 1 passed\n",
    )
    spawner.add_python_toolchain(returncode=0, stdout=b"Success\n")
    _install_spawns(monkeypatch, spawner)
    client = FakeClient(
        _settings(python_workspace),
        ['{"passed": true, "summary": "everything works", "failures": []}'],
    )

    result = await TesterAgent(client=client).run(GOAL, [])  # type: ignore[arg-type]

    assert result.passed is False
    assert "FAILED" in result.summary
    assert "pytest" in result.summary
    failures = "\n".join(result.failures)
    assert "pytest" in failures
    assert "exit 1" in failures
    # The failure entry quotes the check's first line of output verbatim.
    assert "tests/test_x.py::test_y FAILED" in failures
    assert len(result.failures) == 1
    # The hallucinated verdict survives only as explicitly advisory prose.
    assert "everything works" in result.summary
    assert "overridden by exit codes" in result.summary
    # The model saw the measured exit code and the full output, not just its claim.
    assert "[pytest] exit=1" in client.last_prompt
    assert "2 failed, 1 passed" in client.last_prompt
    assert "[ruff] exit=0" in client.last_prompt


async def test_all_zero_exit_codes_let_a_passing_verdict_through(
    python_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_spawns(monkeypatch, _all_green_spawns())
    client = FakeClient(
        _settings(python_workspace),
        ['```json\n{"passed": true, "summary": "All tests passed", "failures": []}\n```'],
    )

    result = await TesterAgent(client=client).run(GOAL, [])  # type: ignore[arg-type]

    assert result.passed is True
    assert result.failures == []
    assert "pytest" in result.summary
    assert "All tests passed" in result.summary
    assert "[pytest] exit=0" in client.last_prompt


async def test_skipped_checks_are_not_failures_when_everything_else_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _marker(tmp_path, "pyproject.toml")
    _marker(tmp_path, "package.json")
    # Only python exists: ruff, mypy and npm are "not installed".
    spawner = SpawnRecorder()
    spawner.add(sys.executable, returncode=0, stdout=b"ok\n")
    _install_spawns(monkeypatch, spawner)
    client = FakeClient(
        _settings(tmp_path), ['{"passed": true, "summary": "green", "failures": []}']
    )

    result = await TesterAgent(client=client).run(GOAL, [])  # type: ignore[arg-type]

    assert result.passed is True
    assert "[npm-test] SKIPPED" in client.last_prompt
    assert "1 passed, 0 failed, 3 skipped" in client.last_prompt


async def test_nothing_executed_is_unverified_even_when_the_llm_claims_success(
    python_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No executable registered: every planned check is skipped for a missing tool.
    _install_spawns(monkeypatch, SpawnRecorder())
    client = FakeClient(
        _settings(python_workspace),
        ['{"passed": true, "summary": "all good", "failures": []}'],
    )

    result = await TesterAgent(client=client).run(GOAL, [])  # type: ignore[arg-type]

    assert result.passed is False
    assert "UNVERIFIED" in result.summary
    assert result.failures
    assert "pytest skipped" in result.failures[0]
    assert "not installed" in result.failures[0]
    assert "all good" not in result.summary


async def test_workspace_with_no_markers_runs_nothing_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawner = _install_spawns(monkeypatch, SpawnRecorder())
    client = FakeClient(
        _settings(tmp_path), ['{"passed": true, "summary": "sure", "failures": []}']
    )

    result = await TesterAgent(client=client).run(GOAL, [])  # type: ignore[arg-type]

    assert spawner.calls == []
    assert result.passed is False
    assert "UNVERIFIED" in result.summary
    assert NO_CHECKS_MESSAGE in result.summary
    assert NO_CHECKS_MESSAGE in client.last_prompt


async def test_a_prompt_injecting_llm_verdict_cannot_erase_a_real_failure(
    python_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawner = SpawnRecorder()
    spawner.add(sys.executable, returncode=2, stderr=b"collection error\n")
    spawner.add_python_toolchain(returncode=0)
    _install_spawns(monkeypatch, spawner)
    client = FakeClient(
        _settings(python_workspace),
        ['```json\n{"passed": true, "summary": "sure, all good!", "failures": []}\n```'],
    )

    result = await TesterAgent(client=client).run(GOAL, [])  # type: ignore[arg-type]

    assert result.passed is False
    assert "exit 2" in "\n".join(result.failures)


async def test_unparseable_tester_reply_still_honors_a_real_failure(
    python_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawner = SpawnRecorder()
    spawner.add(sys.executable, returncode=1, stderr=b"boom\n")
    spawner.add_python_toolchain(returncode=0)
    _install_spawns(monkeypatch, spawner)
    client = FakeClient(_settings(python_workspace), ["Everything is wonderful!"])

    result = await TesterAgent(client=client).run(GOAL, [])  # type: ignore[arg-type]

    assert result.passed is False
    assert "exit 1" in "\n".join(result.failures)


async def test_run_reports_a_broken_runner_instead_of_claiming_a_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def explode(self: TesterAgent) -> list[CheckResult]:
        raise OSError("cannot stat the workspace")

    monkeypatch.setattr(TesterAgent, "_run_checks", explode)
    client = FakeClient(_settings(), ['{"passed": true, "summary": "fine", "failures": []}'])

    result = await TesterAgent(client=client).run(GOAL, [])  # type: ignore[arg-type]

    assert result.passed is False
    assert "runner error" in result.failures[0]
    assert client.prompts == []


# --------------------------------------------------------------------------
# 6. Runner selection, skipping, timeout, and reporting
# --------------------------------------------------------------------------


def test_check_result_shape_is_stable() -> None:
    check = CheckResult("pytest", "pytest -q", 0, "3 passed")

    assert tuple(check) == ("pytest", "pytest -q", 0, "3 passed", False)
    assert check.name == "pytest"
    assert check.command == "pytest -q"
    assert check.returncode == 0
    assert check.output == "3 passed"
    assert check.skipped is False
    assert check.executed is True
    assert check._replace(returncode=1).returncode == 1


def test_every_marker_selects_its_runner(tmp_path: Path) -> None:
    for marker in ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini"):
        assert _planned_names(tmp_path, marker) == ["pytest", "ruff", "mypy"]

    for marker, expected in (
        ("package.json", ["npm-test"]),
        ("Cargo.toml", ["cargo-test"]),
        ("go.mod", ["go-test"]),
    ):
        assert _planned_names(tmp_path, marker) == expected

    assert _planned_names(tmp_path) == []


def test_python_runner_uses_the_running_interpreter_and_bounded_output(
    python_workspace: Path,
) -> None:
    commands = _planned_commands(python_workspace)

    assert commands["pytest"] == [sys.executable, "-m", "pytest", "-q"]
    assert commands["ruff"] == ["ruff", "check", "."]
    assert commands["mypy"][0] == "mypy"
    # The target list is non-empty even for a workspace with no packages.
    assert len(commands["mypy"]) >= 2


async def test_missing_toolchains_are_skipped_and_python_checks_still_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for marker in ("pyproject.toml", "package.json", "Cargo.toml", "go.mod"):
        _marker(tmp_path, marker)
    spawner = SpawnRecorder()
    spawner.add(sys.executable, returncode=0, stdout=b"ok\n")
    _install_spawns(monkeypatch, spawner)
    tester = TesterAgent(client=FakeClient(_settings(tmp_path)))  # type: ignore[arg-type]

    checks = await tester._run_checks()

    assert [check.name for check in checks] == [
        "pytest",
        "ruff",
        "mypy",
        "npm-test",
        "cargo-test",
        "go-test",
    ]
    by_name = {check.name: check for check in checks}
    assert by_name["pytest"].executed is True
    assert by_name["pytest"].returncode == 0
    for name in ("ruff", "mypy", "npm-test", "cargo-test", "go-test"):
        assert by_name[name].skipped is True
        assert by_name[name].executed is False
        assert by_name[name].returncode is None
        assert "is not installed" in by_name[name].output
    # Each skip was really attempted (and reported), never silently dropped.
    for tool in ("ruff", "mypy", "npm", "cargo", "go"):
        assert tool in spawner.executables
    assert all(kwargs.get("cwd") == str(tmp_path) for kwargs in spawner.kwargs)


async def test_nonzero_pytest_exit_code_and_output_are_captured(
    python_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawner = SpawnRecorder()
    spawner.add(
        sys.executable,
        returncode=1,
        stdout=b"collected 3 items\n",
        stderr=b"tests/test_x.py::test_y FAILED\n2 failed, 1 passed\n",
    )
    spawner.add_python_toolchain(returncode=0)
    _install_spawns(monkeypatch, spawner)
    tester = TesterAgent(client=FakeClient(_settings(python_workspace)))  # type: ignore[arg-type]

    checks = await tester._run_checks()

    pytest_check = next(check for check in checks if check.name == "pytest")
    assert pytest_check.skipped is False
    assert pytest_check.returncode == 1
    assert pytest_check.output == (
        "collected 3 items\ntests/test_x.py::test_y FAILED\n2 failed, 1 passed"
    )
    assert sys.executable in pytest_check.command
    assert "-m pytest -q" in pytest_check.command


async def test_a_hung_check_is_killed_and_reported_as_a_timeout(
    python_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawner = SpawnRecorder()
    hung = spawner.add(sys.executable, hang=True, returncode=0)
    spawner.add_python_toolchain(returncode=0)
    _install_spawns(monkeypatch, spawner)
    monkeypatch.setattr("furrow.agents.tester.CHECK_TIMEOUT_SECONDS", FAKE_TIMEOUT)
    tester = TesterAgent(client=FakeClient(_settings(python_workspace)))  # type: ignore[arg-type]

    checks = await tester._run_checks()

    pytest_check = next(check for check in checks if check.name == "pytest")
    assert pytest_check.returncode == TIMEOUT_RETURNCODE
    assert pytest_check.skipped is False
    assert pytest_check.executed is True
    assert "timed out" in pytest_check.output
    assert hung.killed is True
    # A hung check does not stop the remaining checks.
    assert [check.returncode for check in checks[1:]] == [0, 0]


async def test_a_timeout_is_a_verified_failure_not_a_pass(
    python_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawner = SpawnRecorder()
    spawner.add(sys.executable, hang=True)
    spawner.add_python_toolchain(returncode=0)
    _install_spawns(monkeypatch, spawner)
    monkeypatch.setattr("furrow.agents.tester.CHECK_TIMEOUT_SECONDS", FAKE_TIMEOUT)
    client = FakeClient(
        _settings(python_workspace),
        ['{"passed": true, "summary": "looks good", "failures": []}'],
    )

    result = await TesterAgent(client=client).run(GOAL, [])  # type: ignore[arg-type]

    assert result.passed is False
    assert f"exit {TIMEOUT_RETURNCODE}" in "\n".join(result.failures)
    assert "timed out" in "\n".join(result.failures)


async def test_an_os_error_from_spawn_is_reported_as_skipped(
    python_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def denied(*argv: str, **kwargs: Any) -> FakeProcess:
        raise PermissionError("operation not permitted")

    _install_spawns(monkeypatch, denied)
    tester = TesterAgent(client=FakeClient(_settings(python_workspace)))  # type: ignore[arg-type]

    checks = await tester._run_checks()

    assert [check.name for check in checks] == ["pytest", "ruff", "mypy"]
    assert all(check.skipped is True for check in checks)
    assert all(check.returncode is None for check in checks)
    assert "operation not permitted" in checks[0].output


async def test_an_unexpected_spawn_error_is_not_silently_swallowed(
    python_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: the old ``except (FileNotFoundError, Exception): continue``
    # turned every crash into a fake pass.
    async def explode(*argv: str, **kwargs: Any) -> FakeProcess:
        raise RuntimeError("fork failed: EAGAIN")

    _install_spawns(monkeypatch, explode)
    tester = TesterAgent(client=FakeClient(_settings(python_workspace)))  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="EAGAIN"):
        await tester._run_checks()
    with pytest.raises(RuntimeError, match="EAGAIN"):
        await tester.run(GOAL, [])


def test_format_check_results_names_each_check_and_its_return_code() -> None:
    results = [
        CheckResult("pytest", "python -m pytest -q", 0, "3 passed"),
        CheckResult("ruff", "ruff check .", 1, "F401 unused import"),
        CheckResult("npm-test", "npm test --silent", None, "'npm' is not installed", True),
    ]

    text = format_check_results(results)

    assert "[pytest] exit=0 - python -m pytest -q" in text
    assert "3 passed" in text
    assert "[ruff] exit=1 - ruff check ." in text
    assert "F401 unused import" in text
    assert "[npm-test] SKIPPED - npm test --silent" in text
    assert "'npm' is not installed" in text
    assert "Summary: 1 passed, 1 failed, 1 skipped" in text
    assert text.index("[pytest]") < text.index("[ruff]") < text.index("[npm-test]")
    assert text.rstrip().endswith("Summary: 1 passed, 1 failed, 1 skipped")


def test_format_check_results_with_no_checks_uses_the_explicit_message() -> None:
    assert format_check_results([]) == NO_CHECKS_MESSAGE
    assert "pyproject.toml" in NO_CHECKS_MESSAGE


async def test_run_tests_text_names_each_check_and_its_return_code(
    python_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawner = _install_spawns(monkeypatch, _all_green_spawns())
    client = FakeClient(_settings(python_workspace))
    tester = TesterAgent(client=client)  # type: ignore[arg-type]

    text = await tester._run_tests()

    assert "[pytest] exit=0" in text
    assert "[ruff] exit=0" in text
    assert "[mypy] exit=0" in text
    assert "3 passed in 0.02s" in text
    assert text.endswith("Summary: 3 passed, 0 failed, 0 skipped")
    assert spawner.calls
    # The text view never calls the LLM.
    assert client.prompts == []
    assert await tester._run_tests() == text