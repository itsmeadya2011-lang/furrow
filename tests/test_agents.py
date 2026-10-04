"""Regression tests for furrow/agents/{planner,worker,tester,prompts}.py.

Ported from /tmp/kilo/scratch_agents/test_agents_scratch.py and adapted to use the
conftest.py fixtures (fake_llm, settings, tmp_workspace). Every original check is
preserved; no weaker assertions are introduced.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from furrow.agents import MAX_TASKS_PER_PLAN, PlannerAgent, TesterAgent, WorkerAgent
from furrow.agents.prompts import (
    PLANNER_PROMPT,
    TESTER_PROMPT,
    WORKER_PROMPT,
    _schema_hint,
)
from furrow.config import Settings, get_settings
from furrow.models import Plan, TaskModel, TestResult, WorkerResult

VENV_BIN = str(Path(sys.executable).parent)

#: ``TesterAgent`` is a class under test, not a test case. Importing it here makes pytest try to
#: collect it and emit a ``PytestCollectionWarning`` about its ``__init__``. Setting ``__test__``
#: is pytest's supported opt-out, the same treatment :mod:`conftest` gives ``TestResult``.
TesterAgent.__test__ = False  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _runner_binaries_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the venv's bin first on PATH so shutil.which("pytest") resolves."""
    monkeypatch.setenv("PATH", VENV_BIN + os.pathsep + os.environ.get("PATH", ""))


def _settings(workspace: Path, **overrides: Any) -> Settings:
    return Settings(workspace=workspace, **overrides)


# ============================================================ PlannerAgent


FENCED = """Here is the plan you asked for.

```json
{"tasks": [{"id": "1", "description": "add add()", "files": ["app.py"],
            "dependencies": []}], "rationale": "one slice"}
```

Let me know if you want it split further."""


async def test_fenced_and_prose_wrapped_planner_response_parses(
    tmp_workspace: Path, fake_llm
) -> None:
    client = fake_llm(FENCED, settings=_settings(tmp_workspace))
    planner = PlannerAgent(client=client, settings=_settings(tmp_workspace))
    plan = await planner.plan("add add() to app.py")
    assert [t.id for t in plan.tasks] == ["1"]
    assert plan.tasks[0].description == "add add()"
    assert plan.tasks[0].files == ["app.py"]
    assert plan.rationale == "one slice"
    assert str(client.calls[0]["system"]).startswith("You are the planner agent")
    assert client.calls[0]["model"] == _settings(tmp_workspace).planner_model
    assert "Goal:\nadd add() to app.py" in str(client.calls[0]["prompt"])


async def test_nine_tasks_capped_at_five_and_rationale_notes_it(
    tmp_workspace: Path, fake_llm
) -> None:
    body = json.dumps(
        {"tasks": [{"id": str(i), "description": f"task {i}"} for i in range(1, 10)]}
    )
    client = fake_llm(body, settings=_settings(tmp_workspace))
    planner = PlannerAgent(client=client, settings=_settings(tmp_workspace))
    plan = await planner.plan("big goal")
    assert len(plan.tasks) == MAX_TASKS_PER_PLAN == 5
    assert [t.id for t in plan.tasks] == ["1", "2", "3", "4", "5"]
    assert "truncat" in plan.rationale.lower()
    assert "9" in plan.rationale


async def test_ids_normalised_and_dangling_dependencies_dropped(
    tmp_workspace: Path, fake_llm
) -> None:
    body = json.dumps(
        {
            "tasks": [
                {"id": "  ", "description": "blank id"},
                {"id": "a", "description": "first a"},
                {"id": "a", "description": "second a"},
                {"id": "a", "description": "third a"},
                {
                    "description": "no id at all",
                    "dependencies": ["a", "a-2", "ghost"],
                },
            ]
        }
    )
    client = fake_llm(body, settings=_settings(tmp_workspace))
    planner = PlannerAgent(client=client, settings=_settings(tmp_workspace))
    plan = await planner.plan("messy goal")
    assert [t.id for t in plan.tasks] == ["task-1", "a", "a-2", "a-3", "task-5"]
    assert plan.tasks[0].dependencies == []
    assert plan.tasks[4].dependencies == ["a", "a-2"]


async def test_dependencies_on_a_renamed_duplicate_survive(
    tmp_workspace: Path, fake_llm
) -> None:
    body = json.dumps(
        {
            "tasks": [
                {"id": "a", "description": "first"},
                {"id": "a", "description": "second"},
                {"id": "c", "description": "third", "dependencies": ["a"]},
            ]
        }
    )
    client = fake_llm(body, settings=_settings(tmp_workspace))
    planner = PlannerAgent(client=client, settings=_settings(tmp_workspace))
    plan = await planner.plan("d")
    assert [t.id for t in plan.tasks] == ["a", "a-2", "c"]
    assert plan.tasks[2].dependencies == ["a"]


async def test_empty_plan_raises_actionable_value_error(
    tmp_workspace: Path, fake_llm
) -> None:
    client = fake_llm(
        '{"tasks": [], "rationale": "nothing to do"}',
        settings=_settings(tmp_workspace),
    )
    planner = PlannerAgent(client=client, settings=_settings(tmp_workspace))
    with pytest.raises(ValueError) as excinfo:
        await planner.plan("do nothing")
    message = str(excinfo.value)
    assert "no tasks" in message
    assert len(message) < 400


async def test_unparseable_planner_output_is_bounded(
    tmp_workspace: Path, fake_llm
) -> None:
    garbage = "not json at all " * 5000
    client = fake_llm(garbage, settings=_settings(tmp_workspace))
    planner = PlannerAgent(client=client, settings=_settings(tmp_workspace))
    with pytest.raises(ValueError) as excinfo:
        await planner.plan("x")
    message = str(excinfo.value)
    assert "raw response" in message
    assert len(message) < 1500, len(message)
    assert "chars truncated" in message


def test_zero_arg_construction(
    tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FURROW_WORKSPACE", str(tmp_workspace))
    get_settings(refresh=True)
    try:
        planner = PlannerAgent()
        assert planner.settings is not None
        assert planner.client.settings is planner.settings
        worker = WorkerAgent(TaskModel(id="1", description="d"))
        assert worker.client.settings is worker.settings
        assert worker.workspace == tmp_workspace.resolve()
        assert worker.written_files == [] and worker.errors == []
        tester = TesterAgent()
        assert tester.client.settings is tester.settings
        assert tester.workspace == tmp_workspace.resolve()
    finally:
        monkeypatch.delenv("FURROW_WORKSPACE")
        get_settings(refresh=True)


async def test_orchestrator_call_convention(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm,
) -> None:
    monkeypatch.setenv("FURROW_WORKSPACE", str(tmp_workspace))
    get_settings(refresh=True)
    try:
        (tmp_workspace / "main.py").write_text("print('hi')\n", encoding="utf-8")
        body = json.dumps(
            {
                "tasks": [
                    {"id": "1", "description": "print hi", "files": ["main.py"]}
                ],
                "rationale": "one slice",
            }
        )
        client = fake_llm(body, settings=get_settings())
        plan = await PlannerAgent(client=client).plan("make it print hi")
        assert len(plan.tasks) == 1

        worker_body = json.dumps(
            {
                "summary": "done",
                "files": [{"path": "main.py", "content": "print(1)\n"}],
            }
        )
        client2 = fake_llm(worker_body, settings=get_settings())
        summary = await WorkerAgent(task=plan.tasks[0], client=client2).run()
        assert "Wrote: main.py" in summary
        assert (tmp_workspace / "main.py").read_text(encoding="utf-8") == "print(1)\n"
    finally:
        monkeypatch.delenv("FURROW_WORKSPACE")
        get_settings(refresh=True)


# ============================================================ WorkerAgent


async def test_worker_writes_the_file_for_real(
    tmp_workspace: Path, fake_llm
) -> None:
    (tmp_workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    content = "VALUE = 2\n\n\ndef add(a, b):\n    return a + b\n"
    body = json.dumps(
        {
            "summary": "added add()",
            "files": [{"path": "app.py", "content": content}],
            "notes": "runs on 3.10+",
        }
    )
    task = TaskModel(id="1", description="add add()", files=["app.py"])
    client = fake_llm(body, settings=_settings(tmp_workspace))
    worker = WorkerAgent(task=task, client=client, settings=client.settings)

    summary = await worker.run()

    assert (tmp_workspace / "app.py").read_text(encoding="utf-8") == content
    assert worker.written_files == ["app.py"]
    assert "added add()" in summary
    assert "Wrote: app.py" in summary
    assert "Errors:" not in summary
    assert "runs on 3.10+" in summary
    assert task.status == "completed"
    assert task.result == summary
    prompt = str(client.calls[0]["prompt"])
    assert "VALUE = 1" in prompt
    assert "app.py" in prompt
    assert str(client.calls[0]["system"]).startswith("You are a worker agent")


async def test_worker_creates_nested_directories(
    tmp_workspace: Path, fake_llm
) -> None:
    body = json.dumps(
        {"summary": "s", "files": [{"path": "pkg/mod.py", "content": "x = 1\n"}]}
    )
    client = fake_llm(body, settings=_settings(tmp_workspace))
    worker = WorkerAgent(
        task=TaskModel(id="1", description="d"),
        client=client,
        settings=client.settings,
    )
    await worker.run()
    assert (tmp_workspace / "pkg" / "mod.py").read_text(encoding="utf-8") == "x = 1\n"
    assert worker.written_files == ["pkg/mod.py"]


TRAVERSALS = [
    "../escape.py",
    "/etc/passwd",
    "a/../../b.py",
    "~/x.py",
    "C:\\x.py",
]


@pytest.mark.parametrize("bad", TRAVERSALS)
async def test_worker_rejects_traversal(
    tmp_workspace: Path, bad: str, fake_llm
) -> None:
    body = json.dumps(
        {"summary": "sure", "files": [{"path": bad, "content": "pwned\n"}]}
    )
    client = fake_llm(body, settings=_settings(tmp_workspace))
    worker = WorkerAgent(
        task=TaskModel(id="1", description="d"),
        client=client,
        settings=client.settings,
    )
    with pytest.raises(ValueError):
        await worker.run()

    assert worker.written_files == []
    assert len(worker.errors) == 1
    assert worker.errors[0].startswith(f"rejected {bad!r}")
    assert not (tmp_workspace.parent / "escape.py").exists()
    assert not (tmp_workspace.parent / "b.py").exists()
    assert not (tmp_workspace / "a").exists()
    assert not (Path.home() / "x.py").exists()
    assert Path("/etc/passwd").read_text(encoding="utf-8").startswith("root:")


async def test_symlink_escape_is_rejected(
    tmp_workspace: Path, fake_llm
) -> None:
    outside = tmp_workspace.parent / "outside_target.py"
    outside.write_text("original\n", encoding="utf-8")
    (tmp_workspace / "link.py").symlink_to(outside)
    body = json.dumps(
        {"summary": "s", "files": [{"path": "link.py", "content": "clobbered\n"}]}
    )
    client = fake_llm(body, settings=_settings(tmp_workspace))
    worker = WorkerAgent(
        task=TaskModel(id="1", description="d"),
        client=client,
        settings=client.settings,
    )
    with pytest.raises(ValueError):
        await worker.run()
    assert outside.read_text(encoding="utf-8") == "original\n"
    assert "outside the workspace" in worker.errors[0]


async def test_directory_traversal_in_task_files_is_not_read(
    tmp_workspace: Path, fake_llm
) -> None:
    secret = tmp_workspace.parent / "secret.py"
    secret.write_text("TOKEN = 1\n", encoding="utf-8")
    body = json.dumps({"summary": "s", "files": []})
    task = TaskModel(id="1", description="d", files=["../secret.py"])
    client = fake_llm(body, settings=_settings(tmp_workspace))
    worker = WorkerAgent(task=task, client=client, settings=client.settings)
    await worker.run()
    assert "TOKEN" not in str(client.calls[0]["prompt"])
    assert "<unreadable:" in str(client.calls[0]["prompt"])


async def test_delete_and_unknown_action(
    tmp_workspace: Path, fake_llm
) -> None:
    (tmp_workspace / "gone.py").write_text("x\n", encoding="utf-8")
    body = json.dumps(
        {
            "summary": "cleanup",
            "files": [
                {"path": "gone.py", "content": "", "action": "delete"},
                {"path": "missing.py", "content": "", "action": "delete"},
                {"path": "weird.py", "content": "y", "action": "patch"},
            ],
        }
    )
    task = TaskModel(id="1", description="d")
    client = fake_llm(body, settings=_settings(tmp_workspace))
    worker = WorkerAgent(task=task, client=client, settings=client.settings)
    summary = await worker.run()

    assert not (tmp_workspace / "gone.py").exists()
    assert not (tmp_workspace / "missing.py").exists()
    assert not (tmp_workspace / "weird.py").exists()
    assert worker.written_files == []
    assert len(worker.errors) == 1 and "patch" in worker.errors[0]
    assert "Errors:" in summary and task.status == "failed"


async def test_no_changes_is_a_valid_answer(
    tmp_workspace: Path, fake_llm
) -> None:
    body = json.dumps(
        {"summary": "already implemented", "files": [], "notes": "n"}
    )
    task = TaskModel(id="1", description="d")
    client = fake_llm(body, settings=_settings(tmp_workspace))
    worker = WorkerAgent(task=task, client=client, settings=client.settings)
    summary = await worker.run()
    assert summary == "already implemented\nNotes:\nn"
    assert task.status == "completed"
    assert list(tmp_workspace.iterdir()) == []


async def test_fenced_worker_response_parses(
    tmp_workspace: Path, fake_llm
) -> None:
    body = (
        "```json\n"
        + json.dumps(
            {
                "summary": "s",
                "files": [{"path": "a.py", "content": "z\n"}],
            }
        )
        + "\n```"
    )
    client = fake_llm(body, settings=_settings(tmp_workspace))
    worker = WorkerAgent(
        task=TaskModel(id="1", description="d"),
        client=client,
        settings=client.settings,
    )
    await worker.run()
    assert (tmp_workspace / "a.py").read_text(encoding="utf-8") == "z\n"


async def test_unparseable_worker_output_raises(
    tmp_workspace: Path, fake_llm
) -> None:
    task = TaskModel(id="1", description="d")
    client = fake_llm(
        "I would love to help with that!", settings=_settings(tmp_workspace)
    )
    worker = WorkerAgent(task=task, client=client, settings=client.settings)
    with pytest.raises(ValueError) as excinfo:
        await worker.run()
    assert "WorkerResult" in str(excinfo.value)
    assert task.status == "failed"


async def test_worker_partial_failure_does_not_raise(
    tmp_workspace: Path, fake_llm
) -> None:
    body = json.dumps(
        {
            "summary": "partial",
            "files": [
                {"path": "../nope.py", "content": "x"},
                {"path": "ok.py", "content": "fine\n"},
            ],
        }
    )
    task = TaskModel(id="1", description="d")
    client = fake_llm(body, settings=_settings(tmp_workspace))
    worker = WorkerAgent(task=task, client=client, settings=client.settings)
    summary = await worker.run()
    assert (tmp_workspace / "ok.py").read_text(encoding="utf-8") == "fine\n"
    assert worker.written_files == ["ok.py"]
    assert task.status == "failed"
    assert "Errors:" in summary


# ============================================================ TesterAgent


def _project(workspace: Path, body: str) -> None:
    (workspace / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (workspace / "test_math.py").write_text(body, encoding="utf-8")


@pytest.mark.parametrize(
    ("marker", "content", "expected"),
    [
        ("pytest.ini", "[pytest]\n", ["pytest", "-q"]),
        ("tox.ini", "[tox]\n", ["pytest", "-q"]),
        ("pyproject.toml", "[tool.pytest.ini_options]\n", ["pytest", "-q"]),
        ("setup.cfg", "[tool:pytest]\n", ["pytest", "-q"]),
    ],
)
def test_pytest_markers(
    tmp_workspace: Path,
    marker: str,
    content: str,
    expected: list[str],
    fake_llm,
) -> None:
    (tmp_workspace / marker).write_text(content, encoding="utf-8")
    client = fake_llm("{}")
    tester = TesterAgent(client=client, settings=client.settings)
    assert tester.discover_command() == expected


def test_tests_directory_markers(tmp_workspace: Path, fake_llm) -> None:
    client = fake_llm("{}")
    tester = TesterAgent(client=client, settings=client.settings)
    assert tester.discover_command() is None
    tests = tmp_workspace / "tests"
    tests.mkdir()
    assert tester.discover_command() is None
    (tests / "conftest.py").write_text("", encoding="utf-8")
    assert tester.discover_command() == ["pytest", "-q"]
    (tests / "conftest.py").unlink()
    (tests / "test_thing.py").write_text(
        "def test_x():\n    pass\n", encoding="utf-8"
    )
    assert tester.discover_command() == ["pytest", "-q"]
    (tests / "test_thing.py").unlink()
    nested = tests / "unit"
    nested.mkdir()
    (nested / "test_deep.py").write_text(
        "def test_x():\n    pass\n", encoding="utf-8"
    )
    assert tester.discover_command() == ["pytest", "-q"]


def test_npm_marker(tmp_workspace: Path, fake_llm) -> None:
    client = fake_llm("{}")
    tester = TesterAgent(client=client, settings=client.settings)
    (tmp_workspace / "package.json").write_text(
        '{"scripts": {"build": "tsc"}}', encoding="utf-8"
    )
    assert tester.discover_command() is None
    (tmp_workspace / "package.json").write_text(
        '{"scripts": {"test": "jest"}}', encoding="utf-8"
    )
    assert tester.discover_command() == ["npm", "test", "--silent"]
    (tmp_workspace / "package.json").write_text(
        "{not json", encoding="utf-8"
    )
    assert tester.discover_command() is None


def test_cargo_and_go_markers_need_the_binary(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_llm,
) -> None:
    import furrow.agents.tester as tester_module

    real_which = shutil.which
    monkeypatch.setattr(
        tester_module.shutil, "which", lambda name: f"/usr/bin/{name}"
    )

    (tmp_workspace / "Cargo.toml").write_text(
        "[package]\nname='x'\n", encoding="utf-8"
    )
    (tmp_workspace / "go.mod").write_text("module x\n", encoding="utf-8")
    client = fake_llm("{}")
    tester = TesterAgent(client=client, settings=client.settings)
    assert tester.discover_command() == ["cargo", "test", "-q"]
    (tmp_workspace / "Cargo.toml").unlink()
    assert tester.discover_command() == ["go", "test", "./..."]
    (tmp_workspace / "go.mod").unlink()
    assert tester.discover_command() is None

    (tmp_workspace / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    assert tester.discover_command() == ["pytest", "-q"]

    (tmp_workspace / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    monkeypatch.setattr(
        tester_module.shutil,
        "which",
        lambda name: None if name == "pytest" else f"/usr/bin/{name}",
    )
    assert tester.discover_command() == ["cargo", "test", "-q"]
    assert real_which("pytest") is not None


def test_explicit_test_command_wins_and_is_shlex_split(
    tmp_workspace: Path, fake_llm
) -> None:
    (tmp_workspace / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    client = fake_llm("{}", settings=_settings(tmp_workspace, test_command="make t"))
    tester = TesterAgent(client=client, settings=client.settings)
    assert tester.discover_command() == ["make", "t"]
    client = fake_llm(
        "{}",
        settings=_settings(
            tmp_workspace, test_command='python -m pytest -q -k "a b"'
        ),
    )
    tester = TesterAgent(client=client, settings=client.settings)
    assert tester.discover_command() == ["python", "-m", "pytest", "-q", "-k", "a b"]
    client = fake_llm("{}", settings=_settings(tmp_workspace, test_command="   "))
    tester = TesterAgent(client=client, settings=client.settings)
    assert tester.discover_command() is None
    client = fake_llm(
        "{}", settings=_settings(tmp_workspace, test_command='python -c "unbalanced')
    )
    tester = TesterAgent(client=client, settings=client.settings)
    with pytest.raises(ValueError, match="not parseable"):
        tester.discover_command()


async def test_no_suite_is_skipped_and_never_calls_the_model(
    tmp_workspace: Path, fake_llm
) -> None:
    client = fake_llm("{}")
    tester = TesterAgent(client=client, settings=client.settings)
    result = await tester.run("goal", [TaskModel(id="1", description="d")])
    assert result.skipped is True
    assert result.passed is True
    assert result.failures == []
    assert "No runnable test suite detected" in result.summary
    assert str(tmp_workspace) in result.summary
    assert client.calls == []


PASSING = json.dumps(
    {"passed": True, "summary": "4 passed", "failures": [], "skipped": False}
)
FAILING = json.dumps(
    {
        "passed": False,
        "summary": "1 failed",
        "failures": ["tests/test_math.py::test_add expected 3, got 4"],
        "skipped": False,
    }
)


async def test_real_pytest_run_passing(
    tmp_workspace: Path, fake_llm
) -> None:
    _project(tmp_workspace, "def test_add():\n    assert 1 + 1 == 2\n")
    client = fake_llm(PASSING, settings=_settings(tmp_workspace))
    tester = TesterAgent(client=client, settings=client.settings)
    result = await tester.run("goal", [TaskModel(id="1", description="add math")])
    assert result.passed is True and result.skipped is False
    prompt = str(client.calls[0]["prompt"])
    assert "--- stdout ---" in prompt and "--- stderr ---" in prompt
    assert "Exit code: 0" in prompt
    assert "1 passed" in prompt
    assert "add math" in prompt


async def test_real_pytest_run_failing(
    tmp_workspace: Path, fake_llm
) -> None:
    _project(tmp_workspace, "def test_add():\n    assert 1 + 1 == 3\n")
    client = fake_llm(FAILING, settings=_settings(tmp_workspace))
    tester = TesterAgent(client=client, settings=client.settings)
    result = await tester.run("goal", [TaskModel(id="1", description="math")])
    assert result.passed is False
    assert result.skipped is False
    assert result.failures == ["tests/test_math.py::test_add expected 3, got 4"]


async def test_deterministic_fallback_when_the_model_is_useless(
    tmp_workspace: Path, fake_llm
) -> None:
    _project(tmp_workspace, "def test_add():\n    assert 1 + 1 == 3\n")
    client = fake_llm(
        "I think the tests did not pass, probably fine",
        settings=_settings(tmp_workspace),
    )
    tester = TesterAgent(client=client, settings=client.settings)
    result = await tester.run("goal", [TaskModel(id="1", description="math")])
    assert result.passed is False
    assert result.skipped is False
    assert result.failures
    assert "test_math" in " ".join(result.failures)


async def test_fallback_passes_only_on_exit_zero(
    tmp_workspace: Path, fake_llm
) -> None:
    _project(tmp_workspace, "def test_add():\n    assert 1 + 1 == 2\n")
    client = fake_llm(
        "no idea, the tests did not pass",
        settings=_settings(tmp_workspace),
    )
    tester = TesterAgent(client=client, settings=client.settings)
    result = await tester.run("goal", [TaskModel(id="1", description="math")])
    assert result.passed is True
    assert result.failures == []


async def test_model_cannot_claim_skipped_when_a_command_ran(
    tmp_workspace: Path, fake_llm
) -> None:
    _project(tmp_workspace, "def test_add():\n    assert 1 + 1 == 2\n")
    lying = json.dumps(
        {"passed": True, "summary": "s", "failures": [], "skipped": True}
    )
    client = fake_llm(lying, settings=_settings(tmp_workspace))
    tester = TesterAgent(client=client, settings=client.settings)
    result = await tester.run("g", [])
    assert result.skipped is False


async def test_timeout_kills_and_reaps(
    tmp_workspace: Path, fake_llm
) -> None:
    client = fake_llm(
        PASSING,
        settings=_settings(tmp_workspace, test_command="sleep 5", test_timeout=1),
    )
    tester = TesterAgent(client=client, settings=client.settings)
    started = time.monotonic()
    result = await asyncio.wait_for(
        tester.run("goal", [TaskModel(id="1", description="d")]), timeout=8
    )
    elapsed = time.monotonic() - started
    assert elapsed < 4, elapsed
    prompt = str(client.calls[0]["prompt"])
    assert "timed out after 1s" in prompt
    assert "Exit code: 124" in prompt
    assert result.passed is True


async def test_timeout_returns_124_and_leaves_no_orphan(
    tmp_workspace: Path, fake_llm
) -> None:
    sleeper = tmp_workspace / "sleeper.py"
    sleeper.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    try:
        cmd = [sys.executable, str(sleeper)]
        client = fake_llm(
            PASSING,
            settings=_settings(tmp_workspace, test_command="sleep 30", test_timeout=1),
        )
        tester = TesterAgent(client=client, settings=client.settings)
        started = time.monotonic()
        returncode, output = await asyncio.wait_for(
            tester._execute(cmd), timeout=8
        )
        assert returncode == 124
        assert output.startswith("<timed out after 1s>")
        assert time.monotonic() - started < 4
        proc = await asyncio.create_subprocess_exec(
            "pgrep",
            "-f",
            str(sleeper),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        assert proc.returncode == 1, f"orphan left behind: {out!r} {err!r}"
    finally:
        sleeper.unlink(missing_ok=True)


async def test_timeout_keeps_the_partial_output(
    tmp_workspace: Path, fake_llm
) -> None:
    command = shlex.join(
        [
            sys.executable,
            "-c",
            "import time,sys; print('PARTIAL-OUTPUT', flush=True); time.sleep(30)",
        ]
    )
    client = fake_llm(
        PASSING,
        settings=_settings(tmp_workspace, test_command=command, test_timeout=2),
    )
    tester = TesterAgent(client=client, settings=client.settings)
    returncode, output = await tester._execute(shlex.split(command))
    assert returncode == 124
    assert "PARTIAL-OUTPUT" in output


async def test_non_utf8_output_does_not_raise(
    tmp_workspace: Path, fake_llm
) -> None:
    script = "import sys; sys.stdout.buffer.write(b'\\xff\\xfe not utf-8 \\x80\\n')"
    command = shlex.join([sys.executable, "-c", script])
    client = fake_llm(
        PASSING, settings=_settings(tmp_workspace, test_command=command)
    )
    tester = TesterAgent(client=client, settings=client.settings)
    result = await tester.run("goal", [])
    prompt = str(client.calls[0]["prompt"])
    assert "not utf-8" in prompt
    assert result.passed is True


async def test_missing_binary_is_reported_not_raised(
    tmp_workspace: Path, fake_llm
) -> None:
    client = fake_llm(
        PASSING,
        settings=_settings(
            tmp_workspace, test_command="definitely-not-a-binary-xyz"
        ),
    )
    tester = TesterAgent(client=client, settings=client.settings)
    returncode, output = await tester._execute(
        ["definitely-not-a-binary-xyz"]
    )
    assert returncode == 127
    assert "could not start" in output


# ============================================================ Prompts drift guard


def test_tester_output_is_clipped_head_and_tail() -> None:
    from furrow.agents.tester import MAX_OUTPUT_CHARS, _clip

    text = "HEADMARKER\n" + ("x" * 50_000) + "\nTAILMARKER\n"
    clipped = _clip(text, MAX_OUTPUT_CHARS)
    assert len(clipped) <= MAX_OUTPUT_CHARS
    assert clipped.startswith("HEADMARKER")
    assert clipped.rstrip().endswith("TAILMARKER")
    assert "omitted" in clipped
    assert _clip("short", MAX_OUTPUT_CHARS) == "short"


def test_prompt_contents() -> None:
    for prompt, model in (
        (PLANNER_PROMPT, Plan),
        (WORKER_PROMPT, WorkerResult),
        (TESTER_PROMPT, TestResult),
    ):
        assert _schema_hint(model) in prompt
        assert "furrow/models.py" in prompt
    assert "1-5" in PLANNER_PROMPT and "unique" in PLANNER_PROMPT
    assert "never absolute" in WORKER_PROMPT.lower()
    assert ".." in WORKER_PROMPT
    assert "empty `files` list" in WORKER_PROMPT
    assert "self-contained" in TESTER_PROMPT
    assert "skipped" in TESTER_PROMPT
    assert "lint" in TESTER_PROMPT and "type-check" in TESTER_PROMPT


# ============================================================ Package exports


def test_agents_package_exports() -> None:
    import furrow.agents as agents

    assert agents.__all__ == [
        "MAX_TASKS_PER_PLAN",
        "PlannerAgent",
        "TesterAgent",
        "WorkerAgent",
    ]
    for name in agents.__all__:
        assert getattr(agents, name) is not None
    assert agents.__doc__
    from furrow.agents.planner import MAX_TASKS_PER_PLAN as planner_cap

    assert agents.MAX_TASKS_PER_PLAN is planner_cap == 5
