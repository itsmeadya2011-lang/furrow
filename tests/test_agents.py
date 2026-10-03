"""Tests for :mod:`furrow.agents`: planner, worker and tester behaviour.

No LLM is ever contacted: every test drives a local :class:`FakeClient`, and the
handful of tests that really execute a test command use ``sys.executable`` in a
throwaway ``tmp_path`` project with a short timeout.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest

from furrow.agents import tester as tester_mod
from furrow.agents.planner import PlannerAgent, PlannerError
from furrow.agents.tester import NO_RUNNER_OUTPUT
from furrow.agents.worker import MAX_FILE_CHARS, TRUNCATION_MARKER, WorkerAgent
from furrow.config import Plan, Settings, TaskModel
from furrow.config import TestResult as Report  # aliased: not a pytest test class
from furrow.llm import JSONParseError, MissingAPIKeyError, extract_json

# NOTE: the tester agent is deliberately reached as ``tester_mod.TesterAgent``.
# Any module-level alias (Tester, TesterAgent, ...) would start with "Test" and
# pytest would then try to collect the class as a test class.

PYTEST_COMMAND = ["python", "-m", "pytest", "-q"]

# --------------------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ambient ``FURROW_*`` overrides out of the Settings under test."""
    for name in [item for item in os.environ if item.startswith("FURROW_")]:
        monkeypatch.delenv(name, raising=False)


def _settings(**overrides: Any) -> Settings:
    """Isolated ``Settings`` with distinguishable per-agent model names."""
    values: dict[str, Any] = {
        "planner_model": "model-for-planner",
        "worker_model": "model-for-worker",
        "tester_model": "model-for-tester",
        "test_timeout": 30,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class FakeClient:
    """Minimal stand-in for :class:`furrow.llm.LLMClient`.

    Scripted ``str`` replies are run through :func:`extract_json` so a test can feed
    the agents raw model output; any other scripted value is returned verbatim.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        json_replies: list[Any] | None = None,
        text_replies: list[str] | None = None,
        json_error: BaseException | None = None,
        text_error: BaseException | None = None,
        files: dict[str, str] | None = None,
    ) -> None:
        self.settings = settings
        self.json_replies = list(json_replies or [])
        self.text_replies = list(text_replies or [])
        self.json_error = json_error
        self.text_error = text_error
        self.files = dict(files or {})
        self.json_calls: list[dict[str, Any]] = []
        self.text_calls: list[dict[str, Any]] = []
        self.read_paths: list[Path] = []

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        self.json_calls.append({"prompt": prompt, "system": system, "model": model, **kwargs})
        if self.json_error is not None:
            raise self.json_error
        if not self.json_replies:
            raise AssertionError("complete_json called more often than the test scripted")
        reply = self.json_replies.pop(0)
        if isinstance(reply, str):
            return extract_json(reply)
        return reply

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        self.text_calls.append({"prompt": prompt, "system": system, "model": model, **kwargs})
        if self.text_error is not None:
            raise self.text_error
        if not self.text_replies:
            raise AssertionError("complete called more often than the test scripted")
        return self.text_replies.pop(0)

    async def read_file(self, path: str | Path) -> str:
        target = Path(path)
        self.read_paths.append(target)
        if target.name not in self.files:
            raise OSError(f"cannot read {target}")
        return self.files[target.name]


def _fenced(payload: Any) -> str:
    return "Sure, here is the plan:\n```json\n" + json.dumps(payload) + "\n```\nDone."


def _make_project(root: Path, files: dict[str, str], dirs: tuple[str, ...] = ()) -> Path:
    """Materialise a throwaway project tree under ``root``."""
    for name in dirs:
        (root / name).mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def _task_payload(
    task_id: str = "1",
    description: str = "do the thing",
    files: list[str] | None = None,
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": task_id, "description": description}
    if files is not None:
        payload["files"] = files
    if dependencies is not None:
        payload["dependencies"] = dependencies
    return payload


def _assert_acyclic(plan: Plan) -> None:
    graph = {task.id: list(task.dependencies) for task in plan.tasks}
    for task_id in graph:
        seen: set[str] = set()
        pending = list(graph[task_id])
        while pending:
            current = pending.pop()
            assert current not in seen, f"cycle through {current!r}"
            seen.add(current)
            pending.extend(graph.get(current, []))


# ----------------------------------------------------------------------------- planner


async def test_plan_parses_fenced_reply_into_a_plan(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    client = FakeClient(
        settings,
        json_replies=[
            _fenced(
                {
                    "tasks": [
                        _task_payload("1", "write the module", ["src/app.py"]),
                        _task_payload("2", "write the tests", ["tests/test_app.py"], ["1"]),
                    ],
                    "rationale": "slice one",
                }
            )
        ],
    )
    plan = await PlannerAgent(client=client).plan("build an app", "python project")

    assert [task.id for task in plan.tasks] == ["1", "2"]
    assert plan.tasks[0].files == ["src/app.py"]
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.rationale == "slice one"
    call = client.json_calls[0]
    assert call["model"] == "model-for-planner"
    assert "build an app" in call["prompt"]
    assert "python project" in call["prompt"]


async def test_plan_uses_explicit_settings_over_the_client(tmp_path: Path) -> None:
    client = FakeClient(_settings(workspace=tmp_path), json_replies=[{"tasks": []}])
    planner = PlannerAgent(client=client, settings=_settings(planner_model="explicit-model"))
    with pytest.raises(PlannerError):
        await planner.plan("goal")
    assert client.json_calls[0]["model"] == "explicit-model"


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        (["", ""], ["1", "2"]),
        (["", "keep-me"], ["1", "2"]),
        (["dup", "dup"], ["1", "2"]),
        (["dup", "dup", "dup"], ["1", "2", "3"]),
    ],
)
async def test_plan_rewrites_missing_or_duplicated_ids(
    tmp_path: Path, ids: list[str], expected: list[str]
) -> None:
    settings = _settings(workspace=tmp_path)
    tasks = [_task_payload(task_id, f"task {index}") for index, task_id in enumerate(ids)]
    client = FakeClient(settings, json_replies=[{"tasks": tasks}])
    plan = await PlannerAgent(client=client).plan("goal")
    assert [task.id for task in plan.tasks] == expected


async def test_plan_keeps_unique_non_empty_ids(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    tasks = [_task_payload("alpha", "first"), _task_payload("beta", "second")]
    client = FakeClient(settings, json_replies=[{"tasks": tasks}])
    plan = await PlannerAgent(client=client).plan("goal")
    assert [task.id for task in plan.tasks] == ["alpha", "beta"]


async def test_plan_clamps_tasks_to_max_parallel_tasks(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path, max_parallel_tasks=2)
    tasks = [_task_payload(str(index), f"task {index}") for index in range(1, 5)]
    client = FakeClient(settings, json_replies=[{"tasks": tasks}])
    plan = await PlannerAgent(client=client).plan("goal")
    assert len(plan.tasks) == 2
    assert [task.description for task in plan.tasks] == ["task 1", "task 2"]


async def test_plan_with_zero_max_parallel_tasks_keeps_one_task(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path, max_parallel_tasks=0)
    tasks = [_task_payload(str(index), f"task {index}") for index in range(1, 4)]
    client = FakeClient(settings, json_replies=[{"tasks": tasks}])
    plan = await PlannerAgent(client=client).plan("goal")
    assert len(plan.tasks) == 1


async def test_plan_drops_unknown_and_self_dependencies(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    tasks = [
        _task_payload("1", "first", dependencies=["1", "nope", "2", "2", ""]),
        _task_payload("2", "second", dependencies=["nope"]),
    ]
    client = FakeClient(settings, json_replies=[{"tasks": tasks}])
    plan = await PlannerAgent(client=client).plan("goal")
    assert plan.tasks[0].dependencies == ["2"]
    assert plan.tasks[1].dependencies == []
    _assert_acyclic(plan)


async def test_plan_breaks_a_dependency_cycle(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    tasks = [
        _task_payload("1", "first", dependencies=["2"]),
        _task_payload("2", "second", dependencies=["3"]),
        _task_payload("3", "third", dependencies=["1"]),
    ]
    client = FakeClient(settings, json_replies=[{"tasks": tasks}])
    plan = await PlannerAgent(client=client).plan("goal")
    _assert_acyclic(plan)
    assert plan.tasks[2].dependencies == []
    assert sum(len(task.dependencies) for task in plan.tasks) == 2


async def test_plan_defaults_missing_files_and_dependencies(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    client = FakeClient(settings, json_replies=[{"tasks": [_task_payload("1", "bare")]}])
    plan = await PlannerAgent(client=client).plan("goal")
    assert plan.tasks[0].files == []
    assert plan.tasks[0].dependencies == []


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ([{"id": "1", "description": "a list is not a plan"}], "expected a JSON object"),
        ('"a bare JSON string"', "expected a JSON object"),
        ("42", "expected a JSON object"),
        ({"rationale": "no tasks key at all"}, "tasks"),
        ({"tasks": []}, "tasks"),
        ({"tasks": {}}, "tasks"),
    ],
    ids=["list", "string", "number", "no-tasks", "empty-tasks", "tasks-not-a-list"],
)
async def test_plan_raises_on_a_reply_that_is_not_a_plan(
    tmp_path: Path, reply: Any, expected: str
) -> None:
    settings = _settings(workspace=tmp_path)
    client = FakeClient(settings, json_replies=[reply])
    with pytest.raises(PlannerError) as excinfo:
        await PlannerAgent(client=client).plan("goal")
    assert expected in str(excinfo.value)


async def test_plan_propagates_json_parse_error_for_a_non_json_reply(tmp_path: Path) -> None:
    """``complete_json`` owns the repair; a still-unparsable reply reaches the caller."""
    settings = _settings(workspace=tmp_path)
    client = FakeClient(settings, json_replies=["not even a dict"])
    with pytest.raises(JSONParseError):
        await PlannerAgent(client=client).plan("goal")


async def test_plan_raises_when_a_task_is_missing_required_fields(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    client = FakeClient(settings, json_replies=[{"tasks": [{"description": "no id"}]}])
    with pytest.raises(PlannerError) as excinfo:
        await PlannerAgent(client=client).plan("goal")
    assert "plan schema" in str(excinfo.value)


async def test_plan_error_message_truncates_a_huge_reply(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    client = FakeClient(settings, json_replies=[{"rationale": "x" * 5000}])
    with pytest.raises(PlannerError) as excinfo:
        await PlannerAgent(client=client).plan("goal")
    assert len(str(excinfo.value)) < 400


def test_normalize_does_not_mutate_its_input(tmp_path: Path) -> None:
    planner = PlannerAgent(
        client=FakeClient(_settings(workspace=tmp_path)),
        settings=_settings(workspace=tmp_path, max_parallel_tasks=5),
    )
    original = Plan(
        tasks=[
            TaskModel(id="", description="first", files=["a.py"], dependencies=["ghost"]),
            TaskModel(id="", description="second"),
        ],
        rationale="why",
    )
    normalized = planner.normalize(original)

    assert [task.id for task in normalized.tasks] == ["1", "2"]
    assert normalized.tasks[0].dependencies == []
    assert original.tasks[0].id == ""
    assert original.tasks[0].dependencies == ["ghost"]
    assert normalized is not original
    assert normalized.tasks[0] is not original.tasks[0]


def test_normalize_defaults_the_rationale(tmp_path: Path) -> None:
    planner = PlannerAgent(client=FakeClient(_settings(workspace=tmp_path)))
    plan = Plan(tasks=[TaskModel(id="1", description="only")])
    assert planner.normalize(plan).rationale == ""


# ---------------------------------------------------------------------------- worker


async def test_worker_prompt_contains_task_metadata_and_attempt(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    task = TaskModel(
        id="task-7",
        description="refactor the parser",
        files=["src/parser.py", "tests/test_parser.py"],
        dependencies=["1", "2"],
    )
    client = FakeClient(settings, text_replies=["did the work"])
    worker = WorkerAgent(task, client=client)

    assert await worker.run_attempt(2, goal="ship it", context="py3.10") == "did the work"

    call = client.text_calls[0]
    assert call["model"] == "model-for-worker"
    prompt = call["prompt"]
    assert "task-7" in prompt
    assert "Attempt: 2" in prompt
    assert "refactor the parser" in prompt
    assert "src/parser.py, tests/test_parser.py" in prompt
    assert "Dependencies: 1, 2" in prompt
    assert "ship it" in prompt
    assert "py3.10" in prompt


async def test_worker_run_defaults_to_attempt_one(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    client = FakeClient(settings, text_replies=["ok"])
    await WorkerAgent(TaskModel(id="1", description="d"), client=client).run()
    assert "Attempt: 1" in client.text_calls[0]["prompt"]


async def test_worker_prompt_marks_unspecified_files_and_no_dependencies(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    client = FakeClient(settings, text_replies=["ok"])
    await WorkerAgent(TaskModel(id="1", description="d"), client=client).run()
    prompt = client.text_calls[0]["prompt"]
    assert "Files this task owns: unspecified" in prompt
    assert "Dependencies: none" in prompt


async def test_worker_clamps_a_non_positive_attempt_number(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    client = FakeClient(settings, text_replies=["ok", "ok"])
    worker = WorkerAgent(TaskModel(id="1", description="d"), client=client)
    await worker.run_attempt(0)
    await worker.run_attempt(-5)
    assert all("Attempt: 1" in call["prompt"] for call in client.text_calls)


async def test_worker_prompt_includes_existing_file_content(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    _make_project(tmp_path, {"src/parser.py": "TOKEN = 1\n"})
    task = TaskModel(id="1", description="d", files=["src/parser.py", "src/missing.py"])
    client = FakeClient(settings, text_replies=["ok"], files={"parser.py": "TOKEN = 1\n"})
    await WorkerAgent(task, client=client).run()
    prompt = client.text_calls[0]["prompt"]
    assert "--- src/parser.py ---" in prompt
    assert "TOKEN = 1" in prompt
    # The worker checks the real filesystem first, so a file that does not exist is
    # never even handed to the client.
    assert client.read_paths == [tmp_path / "src" / "parser.py"]
    assert "--- src/missing.py ---" not in prompt


async def test_worker_resolves_absolute_file_paths(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path / "somewhere-else")
    absolute = _make_project(tmp_path, {"absolute.py": "ABS = 2\n"}) / "absolute.py"
    task = TaskModel(id="1", description="d", files=[str(absolute)])
    client = FakeClient(settings, text_replies=["ok"], files={"absolute.py": "ABS = 2\n"})
    await WorkerAgent(task, client=client).run()
    assert client.read_paths == [absolute]


async def test_worker_truncates_an_oversized_file(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    body = "x" * (MAX_FILE_CHARS + 500)
    _make_project(tmp_path, {"big.py": body})
    task = TaskModel(id="1", description="d", files=["big.py"])
    client = FakeClient(settings, text_replies=["ok"], files={"big.py": body})
    await WorkerAgent(task, client=client).run()
    prompt = client.text_calls[0]["prompt"]
    marker = TRUNCATION_MARKER.format(shown=MAX_FILE_CHARS, total=len(body))
    assert marker in prompt
    assert "x" * (MAX_FILE_CHARS + 1) not in prompt


async def test_worker_does_not_truncate_a_file_at_the_cap(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    body = "y" * MAX_FILE_CHARS
    _make_project(tmp_path, {"exact.py": body})
    task = TaskModel(id="1", description="d", files=["exact.py"])
    client = FakeClient(settings, text_replies=["ok"], files={"exact.py": body})
    await WorkerAgent(task, client=client).run()
    assert "[truncated" not in client.text_calls[0]["prompt"]


async def test_worker_skips_a_missing_file_without_raising(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    task = TaskModel(id="1", description="d", files=["ghost.py", "also-gone.py"])
    client = FakeClient(settings, text_replies=["ok"])
    assert await WorkerAgent(task, client=client).run() == "ok"
    prompt = client.text_calls[0]["prompt"]
    assert "--- ghost.py ---" not in prompt
    assert "Current content of the files this task owns" not in prompt
    assert client.read_paths == []


async def test_worker_skips_a_file_whose_read_raises(tmp_path: Path) -> None:
    """The file exists on disk but the client cannot read it: skipped, not fatal."""
    settings = _settings(workspace=tmp_path)
    _make_project(tmp_path, {"unreadable.py": "SECRET = 1\n"})
    task = TaskModel(id="1", description="d", files=["unreadable.py"])
    client = FakeClient(settings, text_replies=["ok"])
    assert await WorkerAgent(task, client=client).run() == "ok"
    assert client.read_paths == [tmp_path / "unreadable.py"]
    assert "SECRET" not in client.text_calls[0]["prompt"]


async def test_worker_propagates_llm_error(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path)
    task = TaskModel(id="1", description="d")
    client = FakeClient(settings, text_error=MissingAPIKeyError("ANTHROPIC_API_KEY is not set."))
    with pytest.raises(MissingAPIKeyError):
        await WorkerAgent(task, client=client).run()
    with pytest.raises(MissingAPIKeyError):
        await WorkerAgent(task, client=client).run_attempt(2)


# ----------------------------------------------------------------------------- tester


@pytest.mark.parametrize(
    ("name", "files", "dirs", "expected"),
    [
        ("empty", {}, (), None),
        ("tests-dir", {}, ("tests",), PYTEST_COMMAND),
        ("tests-dir-with-file", {"tests/test_a.py": ""}, (), PYTEST_COMMAND),
        ("pytest-ini", {"pytest.ini": "[pytest]\n"}, (), PYTEST_COMMAND),
        ("tox-ini", {"tox.ini": "[tox]\n"}, (), PYTEST_COMMAND),
        ("setup-cfg", {"setup.cfg": "[metadata]\nname = x\n"}, (), PYTEST_COMMAND),
        (
            "pyproject-with-pytest",
            {"pyproject.toml": '[project]\nname = "x"\n\n[tool.pytest.ini_options]\n'},
            (),
            PYTEST_COMMAND,
        ),
        ("pyproject-without-pytest", {"pyproject.toml": '[project]\nname = "x"\n'}, (), None),
        (
            "package-json-with-test",
            {"package.json": '{"scripts": {"test": "jest"}}'},
            (),
            ["npm", "test", "--", "--silent"],
        ),
        ("package-json-without-test", {"package.json": '{"scripts": {"build": "tsc"}}'}, (), None),
        ("package-json-broken", {"package.json": "{not json"}, (), None),
        ("package-json-no-scripts", {"package.json": "{}"}, (), None),
        ("cargo", {"Cargo.toml": '[package]\nname = "x"\n'}, (), ["cargo", "test", "-q"]),
        ("go-mod", {"go.mod": "module x\n"}, (), ["go", "test", "./..."]),
        (
            "python-marker-wins-over-node",
            {"package.json": '{"scripts": {"test": "jest"}}'},
            ("tests",),
            PYTEST_COMMAND,
        ),
    ],
)
def test_detect_command(
    tmp_path: Path,
    name: str,
    files: dict[str, str],
    dirs: tuple[str, ...],
    expected: list[str] | None,
) -> None:
    workspace = _make_project(tmp_path, files, dirs)
    agent = tester_mod.TesterAgent(client=FakeClient(_settings(workspace=workspace)))
    assert agent.detect_command() == expected


def test_detect_command_ignores_a_project_that_only_has_source(tmp_path: Path) -> None:
    workspace = _make_project(tmp_path, {"src/app.py": "x = 1\n", "README.md": "hi"})
    agent = tester_mod.TesterAgent(client=FakeClient(_settings(workspace=workspace)))
    assert agent.detect_command() is None


def test_explicit_test_command_wins_over_detection(tmp_path: Path) -> None:
    workspace = _make_project(tmp_path, {}, ("tests",))
    settings = _settings(workspace=workspace, test_command="  custom-runner --fast  ")
    agent = tester_mod.TesterAgent(client=FakeClient(settings))
    assert agent._command() == ["custom-runner", "--fast"]


async def test_run_tests_uses_the_configured_workspace_as_cwd(tmp_path: Path) -> None:
    workspace = _make_project(tmp_path, {})
    settings = _settings(
        workspace=workspace,
        test_command=f'{sys.executable} -c "import os; print(os.getcwd())"',
    )
    agent = tester_mod.TesterAgent(client=FakeClient(settings))
    run = await agent.run_tests()
    assert run.found is True
    assert run.returncode == 0
    assert run.output.strip() == str(workspace)


async def test_run_tests_reports_a_missing_command_without_raising(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path, test_command="furrow-no-such-test-runner-xyz")
    agent = tester_mod.TesterAgent(client=FakeClient(settings))
    run = await agent.run_tests()
    assert run.found is False
    assert run.returncode is None
    assert "Could not run" in run.output


async def test_run_tests_without_a_runner_is_marked_not_found(tmp_path: Path) -> None:
    agent = tester_mod.TesterAgent(client=FakeClient(_settings(workspace=tmp_path)))
    run = await agent.run_tests()
    assert run.found is False
    assert run.command == []
    assert run.command_text == ""
    assert run.output == NO_RUNNER_OUTPUT


async def test_command_text_is_shell_quoted(tmp_path: Path) -> None:
    settings = _settings(workspace=tmp_path, test_command=f'{sys.executable} -c "print(1)"')
    agent = tester_mod.TesterAgent(client=FakeClient(settings))
    run = await agent.run_tests()
    assert run.command_text == shlex.join(run.command)
    assert run.command == [sys.executable, "-c", "print(1)"]


@pytest.fixture(scope="session")
def sample_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One throwaway project shared by the three real-subprocess tests."""
    root = tmp_path_factory.mktemp("suite")
    _make_project(
        root,
        {
            "tests/test_ok.py": "def test_passes():\n    assert True\n",
            "tests/test_bad.py": "def test_fails():\n    assert 1 == 2\n",
        },
    )
    return root


@pytest.fixture(scope="session")
def exit_code_command() -> str:
    return f'{sys.executable} -c "raise SystemExit(3)"'


def _failing_command() -> str:
    return f"{sys.executable} -m pytest -q -rf -p no:cacheprovider tests/test_bad.py"


def _passing_command() -> str:
    return f"{sys.executable} -m pytest -q -p no:cacheprovider tests/test_ok.py"


def _quoted(command: str) -> str:
    return shlex.join(shlex.split(command))


async def test_failing_suite_is_reported_as_failed(
    sample_project: Path, exit_code_command: str
) -> None:
    settings = _settings(workspace=sample_project, test_command=_failing_command())
    client = FakeClient(settings, json_replies=[{"summary": "1 failed"}])
    result = await tester_mod.TesterAgent(client=client).run(
        "goal", [TaskModel(id="1", description="d")]
    )

    assert isinstance(result, Report)
    assert result.passed is False
    assert result.returncode not in (None, 0)
    assert result.failures
    assert any("test_fails" in failure for failure in result.failures)
    assert result.command is not None
    assert sys.executable in result.command
    assert "tests/test_bad.py" in result.command


async def test_failing_suite_uses_the_deterministic_summary_when_the_llm_is_down(
    sample_project: Path,
) -> None:
    settings = _settings(workspace=sample_project, test_command=_failing_command())
    client = FakeClient(settings, json_error=MissingAPIKeyError("ANTHROPIC_API_KEY is not set."))
    result = await tester_mod.TesterAgent(client=client).run("goal", [])
    assert result.passed is False
    assert result.failures
    assert result.returncode not in (None, 0)


async def test_passing_suite_is_reported_as_passed(sample_project: Path) -> None:
    settings = _settings(workspace=sample_project, test_command=_passing_command())
    client = FakeClient(settings, json_replies=[{"passed": True, "summary": "all green"}])
    result = await tester_mod.TesterAgent(client=client).run("goal", [])
    assert result.passed is True
    assert result.returncode == 0
    assert result.failures == []
    assert result.summary == "all green"
    assert result.command is not None


async def test_timed_out_command_is_killed_and_fails(tmp_path: Path) -> None:
    settings = _settings(
        workspace=tmp_path,
        test_command=f'{sys.executable} -c "import time; time.sleep(30)"',
        test_timeout=1,
    )
    client = FakeClient(settings, json_error=RuntimeError("llm unavailable"))
    agent = tester_mod.TesterAgent(client=client)
    run = await agent.run_tests()
    assert run.found is True
    assert run.returncode is None
    assert "timed out after 1s and was killed" in run.output

    result = await agent.run("goal", [])
    assert result.returncode is None
    assert result.passed is False
    assert result.summary.startswith("Tests did not report an exit code:")
    assert result.failures == []


async def test_project_without_a_runner_is_skipped_not_blocked(tmp_path: Path) -> None:
    workspace = _make_project(tmp_path, {"src/app.py": "x = 1\n"})
    settings = _settings(workspace=workspace)
    client = FakeClient(settings, json_replies=[{"passed": False, "summary": "no tests"}])
    result = await tester_mod.TesterAgent(client=client).run("goal", [])
    assert result.passed is True
    assert result.summary.startswith("Tests skipped:")
    assert NO_RUNNER_OUTPUT in result.summary
    assert result.failures == []
    assert result.returncode is None
    assert result.command is None


async def test_model_verdict_cannot_flip_a_non_zero_exit_to_passed(
    sample_project: Path, exit_code_command: str
) -> None:
    settings = _settings(workspace=sample_project, test_command=exit_code_command)
    client = FakeClient(
        settings,
        json_replies=[{"passed": True, "summary": "everything is fine", "failures": []}],
    )
    result = await tester_mod.TesterAgent(client=client).run("goal", [])
    assert result.returncode == 3
    assert result.passed is False
    assert result.summary == "everything is fine"
    assert result.failures == ["Test command exited with code 3"]


async def test_model_failures_replace_the_extracted_ones(
    sample_project: Path, exit_code_command: str
) -> None:
    settings = _settings(workspace=sample_project, test_command=exit_code_command)
    client = FakeClient(
        settings,
        json_replies=[{"summary": "it broke", "failures": ["a - AssertionError", "  ", "b"]}],
    )
    result = await tester_mod.TesterAgent(client=client).run("goal", [])
    assert result.passed is False
    assert result.summary == "it broke"
    assert result.failures == ["a - AssertionError", "b"]


async def test_deterministic_summary_when_the_model_gives_no_summary(
    sample_project: Path, exit_code_command: str
) -> None:
    settings = _settings(workspace=sample_project, test_command=exit_code_command)
    client = FakeClient(settings, json_replies=[{"passed": False}])
    result = await tester_mod.TesterAgent(client=client).run("goal", [])
    quoted = shlex.join(shlex.split(exit_code_command))
    assert result.summary == f"Tests failed: {quoted} exited with code 3"


async def test_deterministic_summary_when_the_model_agrees_on_a_pass(
    sample_project: Path,
) -> None:
    settings = _settings(workspace=sample_project, test_command=_passing_command())
    client = FakeClient(settings, json_replies=[{"summary": "   ", "failures": ["noise"]}])
    result = await tester_mod.TesterAgent(client=client).run("goal", [])
    assert result.passed is True
    assert result.failures == ["noise"]
    assert result.summary == f"Tests passed: {_quoted(_passing_command())} exited with code 0"


async def test_tester_prompt_reports_the_command_exit_code_and_tasks(
    sample_project: Path, exit_code_command: str
) -> None:
    settings = _settings(workspace=sample_project, test_command=exit_code_command)
    client = FakeClient(settings, json_replies=[{"summary": "nope"}])
    task = TaskModel(id="4", description="wire the parser")
    await tester_mod.TesterAgent(client=client).run("ship the parser", [task], context="py3.10")
    call = client.json_calls[0]
    assert call["model"] == "model-for-tester"
    prompt = call["prompt"]
    assert "ship the parser" in prompt
    assert "Exit code: 3" in prompt
    assert "- [4] wire the parser" in prompt
    assert "py3.10" in prompt
    assert "Test output (tail):" in prompt


async def test_tester_prompt_lists_no_tasks_when_there_are_none(
    sample_project: Path, exit_code_command: str
) -> None:
    settings = _settings(workspace=sample_project, test_command=exit_code_command)
    client = FakeClient(settings, json_replies=[{"summary": "nope"}])
    await tester_mod.TesterAgent(client=client).run("goal", [])
    assert "- none" in client.json_calls[0]["prompt"]


async def test_model_output_is_truncated_to_the_tail(tmp_path: Path) -> None:
    settings = _settings(
        workspace=tmp_path,
        test_command=f'{sys.executable} -c "print(\'HEAD_MARKER\' + \'abcd\' * 300)"',
        max_output_chars=200,
        test_timeout=30,
    )
    client = FakeClient(settings, json_replies=[{"summary": "ok"}])
    result = await tester_mod.TesterAgent(client=client).run("goal", [])
    full_output = "HEAD_MARKER" + "abcd" * 300 + "\n"
    prompt = client.json_calls[0]["prompt"]
    # The prompt embeds ``command_text``, which itself mentions HEAD_MARKER, so the
    # truncation guarantee is checked on the excerpt alone.
    excerpt = prompt.split("Test output (tail):\n", 1)[1]
    assert result.returncode == 0
    assert result.passed is True
    assert len(excerpt) == 200
    assert excerpt == full_output[-200:]
    assert prompt.endswith(excerpt)


async def test_output_truncation_is_floored_at_two_hundred_chars(tmp_path: Path) -> None:
    settings = _settings(
        workspace=tmp_path,
        test_command=f'{sys.executable} -c "print(\'HEAD_MARKER\' + \'abcd\' * 300)"',
        max_output_chars=10,
        test_timeout=30,
    )
    client = FakeClient(settings, json_replies=[{"summary": "ok"}])
    await tester_mod.TesterAgent(client=client).run("goal", [])
    prompt = client.json_calls[0]["prompt"]
    excerpt = prompt.split("Test output (tail):\n", 1)[1]
    assert len(excerpt) == 200


async def test_ansi_escape_codes_are_stripped_from_failures(tmp_path: Path) -> None:
    escape = chr(27)
    script = (
        f"print('{escape}[31mFAILED tests/test_x.py::test_y{escape}[0m'); raise SystemExit(1)"
    )
    settings = _settings(
        workspace=tmp_path,
        test_command=f'{sys.executable} -c "{script}"',
        test_timeout=30,
    )
    client = FakeClient(settings, json_error=RuntimeError("llm down"))
    result = await tester_mod.TesterAgent(client=client).run("goal", [])
    assert result.passed is False
    assert result.returncode == 1
    assert result.failures == ["FAILED tests/test_x.py::test_y"]
    assert escape not in result.failures[0]


def test_extract_failures_is_capped_and_deduplicated() -> None:
    lines = ["FAILED tests/test_x.py::test_y"] * 5
    lines += [f"AssertionError: nope {index}" for index in range(40)]
    failures = tester_mod.TesterAgent._extract_failures("\n".join(lines))
    assert len(failures) == 20
    assert failures.count("FAILED tests/test_x.py::test_y") == 1