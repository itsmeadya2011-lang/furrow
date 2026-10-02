"""Offline tests for the three furrow agents (planner, worker, tester).

Every test here stays offline: the LLM is a local :class:`FakeClient` and all
filesystem work happens under ``tmp_path``.  The only process these tests ever spawn
is ``sys.executable``, always with ``test_command_timeout`` set to a second or two of
wall clock, and a command only sleeps when a test is specifically exercising the
timeout path.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from furrow.agents import PlannerAgent, PlannerError, TesterAgent, WorkerAgent
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SCHEMA, TESTER_PROMPT
from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult
from furrow.core.events import Event, EventKind

# ``TesterAgent`` is an agent, not a pytest test class, but pytest collects any imported
# class whose name starts with ``Test`` and warns about it.  (``TestResult`` opts out at the
# source, with ``__test__ = False`` on the model itself.)
TesterAgent.__test__ = False


class FakeClient:
    """A minimal stand-in for :class:`furrow.llm.LLMClient`.

    Only the attributes the agents actually touch are implemented, and every call is
    recorded so a test can assert on exactly what went over the wire.

    ``replies`` is popped one entry per :meth:`complete` call.  An entry may be a
    string, a callable taking the prompt, or an exception instance to raise.
    """

    def __init__(self, settings: Settings, replies: list[Any] | None = None) -> None:
        self.settings = settings
        self.replies: list[Any] = list(replies or [])
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.models: list[str | None] = []
        self.list_calls: list[tuple[Path | None, int | None]] = []
        self.listing: list[str] = []
        self.default_listing: list[str] = []
        self.written: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None, **kwargs: Any
    ) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        self.models.append(model)
        if not self.replies:
            return ""
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return reply(prompt)
        return reply

    async def read_file(self, path: str | Path, *, max_bytes: int | None = None) -> str:
        return Path(path).read_text(encoding="utf-8")

    async def write_file(self, path: str | Path, content: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.written.append(str(target))

    def list_files(
        self,
        directory: str | Path | None = None,
        *,
        max_files: int | None = None,
        include_dirs: bool = False,
    ) -> list[str]:
        self.list_calls.append(
            (Path(directory) if directory is not None else None, max_files)
        )
        found = self.listing or self.default_listing
        return list(found[:max_files]) if max_files else list(found)


def _plan_json(tasks: list[dict[str, Any]], rationale: str = "because") -> str:
    """Serialize a plan reply the way a well-behaved model would."""
    return json.dumps({"tasks": tasks, "rationale": rationale})


def _action_json(action: str, **fields: Any) -> str:
    """Serialize one worker action reply."""
    return json.dumps({"thought": f"now I will {action}", "action": action, **fields})


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A sandbox root nested inside ``tmp_path`` so escapes land somewhere harmless."""
    root = tmp_path / "ws"
    root.mkdir()
    return root


@pytest.fixture
def settings(workspace: Path) -> Settings:
    return Settings(
        workspace=workspace,
        planner_max_tasks=5,
        worker_max_turns=4,
        test_command_timeout=5.0,
        max_output_chars=20000,
    )


@pytest.fixture
def client(settings: Settings) -> FakeClient:
    return FakeClient(settings)


@pytest.fixture
def task() -> TaskModel:
    return TaskModel(
        id="w1",
        description="Add parse() to src/config.py so it raises ConfigError on a bad key",
        files=["src/config.py"],
    )


@pytest.fixture
def runners(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``shutil.which`` so test-runner detection is deterministic.

    Detection is gated on both the binary and a project marker, so a test that wants
    to prove the marker gate works must make the binary *look* present.
    """

    def install(*names: str) -> None:
        found = {name: f"/usr/bin/{name}" for name in names}
        monkeypatch.setattr(shutil, "which", found.get)

    return install


@pytest.fixture
def fixed_command(monkeypatch: pytest.MonkeyPatch):
    """Pin the argv the tester will run, so no real suite is ever executed."""

    def install(argv: list[str]) -> list[str]:
        async def detect(self: TesterAgent) -> list[str]:
            return list(argv)

        monkeypatch.setattr(TesterAgent, "detect_command", detect)
        return argv

    return install


# ------------------------------------------------------------------ planner: parsing


async def test_planner_parses_a_plain_json_reply(client: FakeClient) -> None:
    client.replies = [
        _plan_json(
            [
                {
                    "id": "1",
                    "description": "Add parse() to src/config.py",
                    "files": ["src/config.py"],
                    "dependencies": [],
                }
            ],
            "one self-contained module",
        )
    ]
    plan = await PlannerAgent(client=client).plan("Add config parsing")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "Add parse() to src/config.py"
    assert plan.tasks[0].files == ["src/config.py"]
    assert plan.tasks[0].status is TaskStatus.PENDING
    assert plan.rationale == "one self-contained module"
    assert client.calls == 1


async def test_planner_parses_fenced_json_wrapped_in_prose(client: FakeClient) -> None:
    client.replies = [
        "Sure! Here is the plan:\n"
        "```json\n"
        '{"tasks": [{"id": "1", "description": "Write the changelog"}], '
        '"rationale": "fenced reply"}\n'
        "```\n"
        "Let me know if you want anything else."
    ]
    plan = await PlannerAgent(client=client).plan("Update the changelog")
    assert [task.id for task in plan.tasks] == ["1"]
    assert plan.rationale == "fenced reply"
    assert client.calls == 1


async def test_planner_accepts_an_empty_task_list_as_already_satisfied(
    client: FakeClient,
) -> None:
    client.replies = [_plan_json([], "the goal is already met")]
    plan = await PlannerAgent(client=client).plan("Make the build faster")
    assert plan.tasks == []
    assert plan.rationale == "the goal is already met"
    assert client.calls == 1


# ----------------------------------------------------------------- planner: repairing


async def test_planner_retries_exactly_once_and_echoes_the_bad_reply(
    client: FakeClient,
) -> None:
    client.replies = [
        "I am not going to answer in JSON today.",
        _plan_json([{"id": "1", "description": "Do the thing"}]),
    ]
    plan = await PlannerAgent(client=client).plan("goal")
    assert [task.description for task in plan.tasks] == ["Do the thing"]
    assert client.calls == 2
    repair = client.prompts[1]
    assert repair.startswith(client.prompts[0])
    assert "I am not going to answer in JSON today." in repair
    assert "could not be used as a plan" in repair
    assert PLANNER_SCHEMA in repair


async def test_planner_raises_planner_error_after_two_junk_replies(
    client: FakeClient,
) -> None:
    junk = "no json here, just prose. " * 60
    client.replies = [junk, junk]
    with pytest.raises(PlannerError) as excinfo:
        await PlannerAgent(client=client).plan("goal")
    message = str(excinfo.value)
    assert client.calls == 2
    assert "[truncated]" in message
    assert "no json here" in message
    assert len(message) < len(junk) + 200


# ------------------------------------------------------------ planner: normalization


async def test_planner_drops_tasks_without_a_usable_description(client: FakeClient) -> None:
    client.replies = [
        _plan_json(
            [
                {"id": "1", "description": "keep me"},
                {"id": "2", "description": "   "},
                {"id": "3", "description": None},
                {"id": "4"},
                "not even a dict",
            ]
        )
    ]
    plan = await PlannerAgent(client=client).plan("goal")
    assert [task.description for task in plan.tasks] == ["keep me"]


async def test_planner_caps_the_plan_at_planner_max_tasks(
    client: FakeClient, settings: Settings
) -> None:
    settings.planner_max_tasks = 2
    client.replies = [
        _plan_json([{"id": str(index), "description": f"task {index}"} for index in range(6)])
    ]
    plan = await PlannerAgent(client=client, settings=settings).plan("goal")
    assert len(plan.tasks) == 2
    assert [task.id for task in plan.tasks] == ["0", "1"]


async def test_planner_dedupes_descriptions_ignoring_case_and_whitespace(
    client: FakeClient,
) -> None:
    client.replies = [
        _plan_json(
            [
                {"id": "1", "description": "Add a README"},
                {"id": "2", "description": "  add a readme  "},
                {"id": "3", "description": "ADD A README\n"},
                {"id": "4", "description": "Add a  README"},
                {"id": "5", "description": "Add\ta readme"},
                {"id": "6", "description": "Add\na readme"},
                {"id": "7", "description": "Add something else"},
            ]
        )
    ]
    plan = await PlannerAgent(client=client).plan("goal")
    # Descriptions are fingerprinted on lowercased, single-spaced text, so every spelling
    # of "add a readme" collapses into the first one the model wrote.
    assert [task.description for task in plan.tasks] == [
        "Add a README",
        "Add something else",
    ]
    assert [task.id for task in plan.tasks] == ["1", "7"]


async def test_planner_repairs_missing_and_duplicate_ids(client: FakeClient) -> None:
    client.replies = [
        _plan_json(
            [
                {"id": "dup", "description": "first"},
                {"id": "dup", "description": "second"},
                {"description": "third"},
            ]
        )
    ]
    plan = await PlannerAgent(client=client).plan("goal")
    assert [task.id for task in plan.tasks] == ["dup", "2", "3"]
    assert len({task.id for task in plan.tasks}) == 3


async def test_planner_coerces_files_and_dependencies_to_string_lists(
    client: FakeClient,
) -> None:
    client.replies = [
        _plan_json(
            [
                {"id": "1", "description": "null fields", "files": None, "dependencies": "no"},
                {
                    "id": "2",
                    "description": "mixed fields",
                    "files": [" a.py ", "", 3, None, True],
                    "dependencies": 7,
                },
            ]
        )
    ]
    plan = await PlannerAgent(client=client).plan("goal")
    assert plan.tasks[0].files == []
    assert plan.tasks[0].dependencies == []
    assert plan.tasks[1].files == ["a.py", "3"]
    assert plan.tasks[1].dependencies == []


async def test_planner_drops_unknown_duplicate_and_self_dependencies(
    client: FakeClient,
) -> None:
    client.replies = [
        _plan_json(
            [
                {"id": "1", "description": "first"},
                {"id": "2", "description": "second", "dependencies": ["1", "1", "2", "ghost"]},
            ]
        )
    ]
    plan = await PlannerAgent(client=client).plan("goal")
    assert plan.tasks[1].dependencies == ["1"]


async def test_planner_defaults_a_missing_rationale(client: FakeClient) -> None:
    client.replies = ['{"tasks": [{"id": "1", "description": "only a task"}]}']
    plan = await PlannerAgent(client=client).plan("goal")
    assert plan.rationale == ""
    assert [task.description for task in plan.tasks] == ["only a task"]


# -------------------------------------------------------------- planner: the prompt


async def test_planner_prompt_carries_the_goal_focus_and_context(
    client: FakeClient,
) -> None:
    client.replies = [_plan_json([], "nothing to do")]
    await PlannerAgent(client=client).plan(
        "Make the web UI show progress",
        focus="tests/test_web.py::test_stream failed",
        context="Previous cycle findings: 1 task failed",
    )
    prompt = client.prompts[0]
    assert "Make the web UI show progress" in prompt
    assert "tests/test_web.py::test_stream failed" in prompt
    assert "Previous cycle findings: 1 task failed" in prompt
    assert "Focus for this cycle" in prompt
    assert "Repository context:" in prompt


async def test_planner_focus_is_framed_as_the_previous_cycle_findings(
    client: FakeClient,
) -> None:
    client.replies = [_plan_json([], "nothing to do")]
    await PlannerAgent(client=client).plan("goal", focus="tests/test_a.py failed")
    prompt = client.prompts[0]
    assert "Focus for this cycle (what is still broken or missing):" in prompt
    assert "tests/test_a.py failed" in prompt


async def test_planner_omits_the_focus_section_when_no_focus_is_given(
    client: FakeClient,
) -> None:
    client.replies = [_plan_json([], "nothing to do")]
    await PlannerAgent(client=client).plan("just a goal")
    prompt = client.prompts[0]
    assert "Focus for this cycle" not in prompt
    assert "Repository context:" not in prompt
    assert "Goal:\njust a goal" in prompt


async def test_planner_truncates_a_long_repository_listing(client: FakeClient) -> None:
    client.default_listing = [
        f"src/very/deeply/nested/package/subpackage/module_number_{index:03d}_pad.py"
        for index in range(40)
    ]
    client.replies = [_plan_json([], "nothing to do")]
    await PlannerAgent(client=client).plan("goal")
    prompt = client.prompts[0]
    assert "Repository layout (truncated):" in prompt
    assert "[truncated]" in prompt
    layout = prompt.split("Repository layout (truncated):\n", 1)[1]
    assert len(layout) <= 2000


async def test_planner_requests_only_a_capped_number_of_files(client: FakeClient) -> None:
    client.default_listing = [f"src/module_{index:03d}.py" for index in range(200)]
    client.replies = [_plan_json([], "nothing to do")]
    await PlannerAgent(client=client).plan("goal")
    directory, max_files = client.list_calls[0]
    assert directory is None
    assert max_files == 40
    assert "src/module_000.py" in client.prompts[0]
    assert "src/module_199.py" not in client.prompts[0]


async def test_planner_sends_its_prompt_as_a_system_argument(client: FakeClient) -> None:
    client.replies = [_plan_json([], "nothing to do")]
    await PlannerAgent(client=client).plan("goal")
    assert client.systems == [PLANNER_PROMPT]
    assert client.models == [client.settings.planner_model]
    assert PLANNER_PROMPT not in client.prompts[0]


# ---------------------------------------------------------------------- worker: loop


async def test_worker_happy_path_writes_the_file_and_returns_the_summary(
    client: FakeClient, task: TaskModel, workspace: Path
) -> None:
    (workspace / "src").mkdir()
    (workspace / "src" / "config.py").write_text("old contents\n", encoding="utf-8")
    client.replies = [
        _action_json("read", path="src/config.py"),
        _action_json("write", path="src/config.py", content="new contents\n"),
        _action_json("finish", summary="Rewrote src/config.py with parse()."),
    ]
    worker = WorkerAgent(task, client=client)
    summary = await worker.run()
    assert summary == "Rewrote src/config.py with parse()."
    assert (workspace / "src" / "config.py").read_text(encoding="utf-8") == "new contents\n"
    assert worker.files_read == ["src/config.py"]
    assert worker.files_written == ["src/config.py"]
    assert worker.errors == []
    assert worker.turns == 3
    assert [record["action"] for record in worker.actions] == ["read", "write", "finish"]
    assert all(record["ok"] for record in worker.actions)
    assert worker.actions[1]["path"] == "src/config.py"


async def test_worker_list_feeds_the_listing_back_to_the_model(
    client: FakeClient, task: TaskModel
) -> None:
    client.listing = ["a.txt", "b.txt"]
    client.replies = [
        _action_json("list"),
        _action_json("finish", summary="Looked around."),
    ]
    worker = WorkerAgent(task, client=client)
    summary = await worker.run()
    assert summary == "Looked around."
    assert client.list_calls == [(None, 60)]
    assert worker.actions[0]["ok"] is True
    assert "[list .]" in client.prompts[1]
    assert "b.txt" in client.prompts[1]


async def test_worker_turn_prompt_names_the_task_and_its_declared_files(
    client: FakeClient, task: TaskModel
) -> None:
    client.replies = [_action_json("finish", summary="Nothing to do.")]
    worker = WorkerAgent(task, client=client)
    await worker.run()
    prompt = client.prompts[0]
    assert "Task id: w1" in prompt
    assert task.description in prompt
    assert "Declared files: src/config.py" in prompt
    assert "Repository root:" in prompt
    assert "Turn 1 of 4" in prompt
    assert client.models == [client.settings.worker_model]


async def test_worker_prompt_says_so_when_no_files_are_declared(
    client: FakeClient,
) -> None:
    client.replies = [_action_json("finish", summary="Nothing to do.")]
    await WorkerAgent(TaskModel(id="w2", description="Find the bug"), client=client).run()
    assert "Declared files: none declared - choose them yourself" in client.prompts[0]


# ------------------------------------------------------------------ worker: sandboxing


@pytest.mark.parametrize("escape", ["../outside.txt", "/etc/outside.txt"])
async def test_worker_refuses_a_write_outside_the_workspace(
    client: FakeClient, task: TaskModel, workspace: Path, escape: str
) -> None:
    client.replies = [
        _action_json("write", path=escape, content="should never land"),
        _action_json("finish", summary="Finished after the refused write."),
    ]
    worker = WorkerAgent(task, client=client)
    summary = await worker.run()
    probe = Path(escape) if escape.startswith("/") else workspace.parent / escape
    assert summary == "Finished after the refused write."
    assert not probe.exists()
    assert worker.files_written == []
    assert worker.actions[0]["ok"] is False
    assert any("escapes root" in error for error in worker.errors)
    assert "should never land" not in client.prompts[1]


async def test_worker_refuses_to_read_outside_the_workspace(
    client: FakeClient, task: TaskModel, workspace: Path
) -> None:
    (workspace.parent / "secret.txt").write_text("top secret", encoding="utf-8")
    client.replies = [
        _action_json("read", path="../secret.txt"),
        _action_json("finish", summary="Gave up on the secret."),
    ]
    worker = WorkerAgent(task, client=client)
    summary = await worker.run()
    assert summary == "Gave up on the secret."
    assert worker.files_read == []
    assert worker.actions[0]["ok"] is False
    assert any("escapes root" in error for error in worker.errors)
    assert "top secret" not in client.prompts[1]


# ------------------------------------------------------------------- worker: bad turns


async def test_worker_returns_a_prose_reply_as_the_summary(
    client: FakeClient, task: TaskModel
) -> None:
    client.replies = ["I read the file and everything already looks correct."]
    worker = WorkerAgent(task, client=client)
    summary = await worker.run()
    assert summary == "I read the file and everything already looks correct."
    assert worker.turns == 1
    assert worker.actions == []
    assert worker.errors == []


async def test_worker_records_an_unknown_action_and_keeps_going(
    client: FakeClient, task: TaskModel
) -> None:
    client.replies = [
        _action_json("delete", path="src/config.py"),
        _action_json("finish", summary="Recovered from the bad action."),
    ]
    worker = WorkerAgent(task, client=client)
    summary = await worker.run()
    assert summary == "Recovered from the bad action."
    assert worker.turns == 2
    assert [record["action"] for record in worker.actions] == ["delete", "finish"]
    assert any("unknown or missing action 'delete'" in error for error in worker.errors)
    assert "unknown or missing action 'delete'" in client.prompts[1]


@pytest.mark.parametrize("fields", [{}, {"content": 123}, {"content": None}])
async def test_worker_refuses_a_write_without_string_content(
    client: FakeClient, task: TaskModel, workspace: Path, fields: dict[str, Any]
) -> None:
    target = workspace / "new.txt"
    target.write_text("original\n", encoding="utf-8")
    client.replies = [
        _action_json("write", path="new.txt", **fields),
        _action_json("finish", summary="Stopped before writing."),
    ]
    worker = WorkerAgent(task, client=client)
    summary = await worker.run()
    assert summary == "Stopped before writing."
    assert target.read_text(encoding="utf-8") == "original\n"
    assert worker.files_written == []
    assert worker.actions[0]["ok"] is False
    assert any("write new.txt action requires a 'content'" in e for e in worker.errors)


async def test_worker_records_a_provider_failure_instead_of_raising(
    client: FakeClient, task: TaskModel
) -> None:
    client.replies = [RuntimeError("provider exploded")]
    worker = WorkerAgent(task, client=client)
    summary = await worker.run()
    assert "model call failed: provider exploded" in worker.errors[0]
    assert "provider exploded" in summary
    assert worker.turns == 1
    assert worker.files_written == []
    assert client.calls == 1


# ----------------------------------------------------------------- worker: turn budget


async def test_worker_stops_at_the_turn_limit(
    client: FakeClient, task: TaskModel, settings: Settings, workspace: Path
) -> None:
    settings.worker_max_turns = 3
    (workspace / "notes.txt").write_text("hello\n", encoding="utf-8")
    client.replies = [_action_json("read", path="notes.txt")] * 3
    worker = WorkerAgent(task, client=client, settings=settings)
    summary = await worker.run()
    assert worker.turns == 3
    assert client.calls == 3
    assert "3-turn limit" in summary
    assert "incomplete" in summary
    assert worker.files_read == ["notes.txt"] * 3
    assert worker.errors == []


async def test_worker_propagates_cancellation(client: FakeClient, task: TaskModel) -> None:
    client.replies = [asyncio.CancelledError()]
    worker = WorkerAgent(task, client=client)
    with pytest.raises(asyncio.CancelledError):
        await worker.run()
    assert worker.turns == 1
    assert worker.errors == []


# --------------------------------------------------------------------- worker: events


async def test_worker_emits_a_log_event_for_a_successful_action(
    client: FakeClient, task: TaskModel
) -> None:
    collected: list[Event] = []

    async def collect(event: Event) -> None:
        collected.append(event)

    client.replies = [
        _action_json("write", path="src/config.py", content="x = 1\n"),
        _action_json("finish", summary="Wrote the module."),
    ]
    worker = WorkerAgent(task, client=client)
    await worker.run(on_event=collect)
    assert len(collected) == 1
    event = collected[0]
    assert event.kind is EventKind.LOG
    assert event.task_id == task.id
    assert event.data["action"] == "write"
    assert event.data["files"] == 1
    assert event.data["path"] == "src/config.py"
    assert event.message == "wrote src/config.py"


async def test_worker_runs_without_an_event_callback(
    client: FakeClient, task: TaskModel, workspace: Path
) -> None:
    client.replies = [
        _action_json("write", path="a.txt", content="hi\n"),
        _action_json("finish", summary="Wrote a.txt."),
    ]
    worker = WorkerAgent(task, client=client)
    summary = await worker.run(on_event=None)
    assert summary == "Wrote a.txt."
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "hi\n"
    assert worker.files_written == ["a.txt"]


# ------------------------------------------------------------------ tester: detection


async def test_detect_command_returns_none_in_an_empty_workspace(
    client: FakeClient, settings: Settings, runners
) -> None:
    runners("pytest", "python", "python3", "npm", "cargo", "go")
    tester = TesterAgent(client=client, settings=settings)
    assert await tester.detect_command() is None


@pytest.mark.parametrize("marker", ["pyproject.toml", "pytest.ini", "setup.py", "conftest.py"])
async def test_detect_command_selects_pytest_for_a_python_marker(
    client: FakeClient, settings: Settings, workspace: Path, runners, marker: str
) -> None:
    runners("pytest")
    (workspace / marker).write_text("[project]\nname = 'x'\n", encoding="utf-8")
    cmd = await TesterAgent(client=client, settings=settings).detect_command()
    assert cmd == ["pytest", "-q"]


async def test_detect_command_selects_pytest_for_a_tests_directory(
    client: FakeClient, settings: Settings, workspace: Path, runners
) -> None:
    runners("pytest")
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_x.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    cmd = await TesterAgent(client=client, settings=settings).detect_command()
    assert cmd == ["pytest", "-q"]


async def test_detect_command_falls_back_to_the_interpreter_without_a_pytest_binary(
    client: FakeClient, settings: Settings, workspace: Path, runners
) -> None:
    runners("python3")
    (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    cmd = await TesterAgent(client=client, settings=settings).detect_command()
    assert cmd == ["/usr/bin/python3", "-m", "pytest", "-q"]


async def test_detect_command_selects_npm_for_package_json(
    client: FakeClient, settings: Settings, workspace: Path, runners
) -> None:
    runners("npm")
    (workspace / "package.json").write_text('{"scripts": {"test": "vitest"}}\n', encoding="utf-8")
    cmd = await TesterAgent(client=client, settings=settings).detect_command()
    assert cmd == ["npm", "test", "--", "--silent"]


async def test_detect_command_selects_cargo_for_a_cargo_project(
    client: FakeClient, settings: Settings, workspace: Path, runners
) -> None:
    runners("cargo")
    (workspace / "Cargo.toml").write_text('[package]\nname = "x"\n', encoding="utf-8")
    cmd = await TesterAgent(client=client, settings=settings).detect_command()
    assert cmd == ["cargo", "test", "-q"]


async def test_detect_command_selects_go_for_a_go_module(
    client: FakeClient, settings: Settings, workspace: Path, runners
) -> None:
    runners("go")
    (workspace / "go.mod").write_text("module example\n", encoding="utf-8")
    cmd = await TesterAgent(client=client, settings=settings).detect_command()
    assert cmd == ["go", "test", "./..."]


# -------------------------------------------------------------------- tester: results


async def test_run_without_a_runner_reports_verification_skipped(
    client: FakeClient, settings: Settings, runners
) -> None:
    runners("pytest", "python", "python3")
    result = await TesterAgent(client=client, settings=settings).run("goal", [])
    assert isinstance(result, TestResult)
    assert result.command is None
    assert result.exit_code is None
    assert result.passed is True
    assert "No test runner detected" in result.summary
    assert client.calls == 0


async def test_run_reports_a_passing_suite_without_calling_the_model(
    client: FakeClient, settings: Settings, fixed_command
) -> None:
    argv = fixed_command([sys.executable, "-c", "print('2 passed in 0.01s')"])
    result = await TesterAgent(client=client, settings=settings).run("goal", [])
    assert result.passed is True
    assert result.exit_code == 0
    assert result.command == " ".join(argv)
    assert result.failures == []
    assert result.summary.startswith("Tests passed:")
    assert "2 passed in 0.01s" in result.summary
    assert client.calls == 0


async def test_run_derives_failures_from_the_output_when_the_reply_is_unusable(
    client: FakeClient, settings: Settings, fixed_command, task: TaskModel
) -> None:
    script = "\n".join(
        [
            "print('tests/test_a.py::test_one FAILED')",
            "print('tests/test_a.py::test_one FAILED')",
            "print('tests/test_b.py::test_two - AssertionError: expected 201, got 500')",
            "print('Error: fixture blew up')",
            "raise SystemExit(1)",
        ]
    )
    fixed_command([sys.executable, "-c", script])
    client.replies = ["I am afraid I cannot help with that."]
    result = await TesterAgent(client=client, settings=settings).run("goal", [task])
    assert result.passed is False
    assert result.exit_code == 1
    assert result.failures == [
        "tests/test_a.py::test_one FAILED",
        "tests/test_b.py::test_two - AssertionError: expected 201, got 500",
        "Error: fixture blew up",
    ]
    assert len(result.failures) == len(set(result.failures))
    assert "AssertionError" in result.summary
    assert client.calls == 1
    assert task.description in client.prompts[0]
    assert client.systems == [TESTER_PROMPT]


async def test_run_treats_the_exit_code_as_the_verdict_not_the_model(
    client: FakeClient, settings: Settings, fixed_command
) -> None:
    fixed_command([sys.executable, "-c", "raise SystemExit(2)"])
    client.replies = [
        json.dumps({"passed": True, "summary": "everything is fine", "failures": []})
    ]
    result = await TesterAgent(client=client, settings=settings).run("goal", [])
    assert result.passed is False
    assert result.exit_code == 2
    assert result.summary == "everything is fine"
    assert result.failures == []


async def test_run_keeps_the_failures_the_model_reports(
    client: FakeClient, settings: Settings, fixed_command
) -> None:
    fixed_command([sys.executable, "-c", "raise SystemExit(1)"])
    client.replies = [
        json.dumps(
            {
                "passed": False,
                "summary": "1 of 2 tests failed",
                "failures": ["tests/test_a.py::test_one - AssertionError: expected 1, got 2"],
            }
        )
    ]
    result = await TesterAgent(client=client, settings=settings).run("goal", [])
    assert result.passed is False
    assert result.exit_code == 1
    assert result.summary == "1 of 2 tests failed"
    assert result.failures == ["tests/test_a.py::test_one - AssertionError: expected 1, got 2"]


# ------------------------------------------------------------------ tester: processes


async def test_run_command_reports_a_hang_as_exit_124(
    client: FakeClient, settings: Settings
) -> None:
    settings.test_command_timeout = 1
    code, output = await TesterAgent(client=client, settings=settings).run_command(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    assert code == 124
    assert "timed out after 1s" in output


async def test_run_reports_a_timed_out_suite_as_a_failure(
    client: FakeClient, settings: Settings, fixed_command
) -> None:
    settings.test_command_timeout = 1
    fixed_command([sys.executable, "-c", "import time; time.sleep(30)"])
    client.replies = ["no json from me"]
    result = await TesterAgent(client=client, settings=settings).run("goal", [])
    assert result.passed is False
    assert result.exit_code == 124
    assert "timed out" in result.summary
    assert result.failures == []


async def test_run_reports_a_missing_runner_binary_as_exit_127(
    client: FakeClient, settings: Settings, fixed_command
) -> None:
    fixed_command(["furrow-no-such-test-runner-xyz"])
    client.replies = ["I could not read that."]
    result = await TesterAgent(client=client, settings=settings).run("goal", [])
    assert result.passed is False
    assert result.exit_code == 127
    assert result.command == "furrow-no-such-test-runner-xyz"
    assert "could not run" in result.summary
    assert result.failures == []


async def test_run_command_truncates_output_to_max_output_chars(
    client: FakeClient, settings: Settings
) -> None:
    settings.max_output_chars = 200
    code, output = await TesterAgent(client=client, settings=settings).run_command(
        [sys.executable, "-c", "print('x' * 5000)"]
    )
    assert code == 0
    assert len(output) == 200
    assert "[truncated]" in output
