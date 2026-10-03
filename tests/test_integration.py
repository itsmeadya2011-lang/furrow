"""End-to-end proof that furrow actually does its job, in one hermetic test file.

What is real here and what is not:

* Real: :class:`~furrow.core.orchestrator.Orchestrator`, the planner's JSON
  normalisation, the scheduler's dependency waves, the **entire unmocked
  ``WorkerAgent`` apply path** (``resolve_write_path`` -> ``create_backup`` ->
  ``atomic_write_text`` -> ``os.replace``) and real files on disk under
  ``tmp_path``.
* Faked: the LLM transport (:class:`FakeClient`, canned replies keyed by model
  name) and ``TesterAgent.run``. The tester is stubbed at the seam the
  orchestrator calls so a verdict can be scripted deterministically; one test
  below exercises the *real* ``TesterAgent.run`` with the command layer stubbed
  instead, so both halves are covered.
* Never used: the network, API keys, the real project test suite, the wall clock
  (every timestamp assertion globs for the backup directory instead of computing
  it), or ``Path.cwd()`` as a write target.

The whole file runs in a couple of seconds and spawns no subprocesses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from furrow.agents import tester as tester_module
from furrow.agents.worker import (
    ACTION_CREATED,
    ACTION_MODIFIED,
    ACTION_SKIPPED,
    WorkerAgent,
)
from furrow.config import Settings, TaskModel, TestResult
from furrow.core.orchestrator import MAX_IDENTICAL_FAILURES, Orchestrator
from furrow.llm import LLMClient

GOAL = "add a health check endpoint"

# Distinct per-role model names so FakeClient can dispatch on them; the real
# Settings defaults are provider-specific strings that would be ambiguous.
PLANNER_MODEL = "fake-planner-model"
WORKER_MODEL = "fake-worker-model"
TESTER_MODEL = "fake-tester-model"


# --------------------------------------------------------------------------- #
# Canned-LLM plumbing
# --------------------------------------------------------------------------- #


class FakeClient(LLMClient):
    """A real ``LLMClient`` with only the transport replaced.

    Subclassing (rather than duck-typing) keeps ``read_file``/``list_files`` real,
    which matters: those are how the worker grounds its prompt in the workspace,
    so the tests exercise the real context-building path rather than a mock of it.
    Reaching a provider is a test bug and fails loudly.
    """

    def __init__(
        self,
        settings: Settings,
        plans: list[str],
        edits: list[str],
        verdicts: list[str] | None = None,
        call_budget: int = 40,
    ) -> None:
        super().__init__(settings=settings)
        self._plans = list(plans)
        self._edits = list(edits)
        self._verdicts = list(verdicts or [])
        self.plan_calls = 0
        self.worker_calls = 0
        self.verdict_calls = 0
        self.prompts: list[tuple[str, str]] = []
        # A runaway loop must fail these tests, never hang the suite. Once a role
        # has been called this many times, complete() raises; three consecutive
        # planner failures trip the orchestrator's own planner guard, so a broken
        # loop stops with a diagnosis instead of spinning forever. No healthy test
        # comes close to the budget (the widest uses four worker calls).
        self._call_budget = call_budget

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append((model or "", prompt))
        queue: list[str] | None = None
        counter = ""
        if model == self.settings.planner_model:
            queue, counter = self._plans, "plan_calls"
        elif model == self.settings.worker_model:
            queue, counter = self._edits, "worker_calls"
        elif model == self.settings.tester_model:
            queue, counter = self._verdicts, "verdict_calls"
        else:
            raise AssertionError(f"unexpected model requested: {model!r}")
        index = getattr(self, counter)
        setattr(self, counter, index + 1)
        if index >= self._call_budget:
            raise AssertionError(
                f"the loop called the {model!r} role {index + 1} times; it is not terminating"
            )
        if not queue:
            raise AssertionError(
                f"no canned reply queued for model {model!r}; the test asked for a real "
                "LLM call, which is forbidden here"
            )
        return queue[min(index, len(queue) - 1)]  # the last reply repeats forever

    def prompts_for(self, model: str) -> list[str]:
        return [prompt for requested, prompt in self.prompts if requested == model]

    @property
    def anthropic(self) -> Any:
        raise AssertionError("FakeClient must not build a real Anthropic client")

    @property
    def openai(self) -> Any:
        raise AssertionError("FakeClient must not build a real OpenAI client")


def make_settings(workspace: Path, **overrides: Any) -> Settings:
    """Settings for one test: fake model names, a quiet logger, a temp workspace."""
    base: dict[str, Any] = {
        "planner_model": PLANNER_MODEL,
        "worker_model": WORKER_MODEL,
        "tester_model": TESTER_MODEL,
        "log_level": "CRITICAL",
        "workspace": workspace,
    }
    base.update(overrides)
    return Settings(**base)


def plan_json(
    *tasks: tuple[str, str, list[str], list[str]], rationale: str = "integration plan"
) -> str:
    """A canned planner reply. Each task tuple is (id, description, files, deps)."""
    return json.dumps(
        {
            "tasks": [
                {
                    "id": task_id,
                    "description": description,
                    "files": files,
                    "dependencies": deps,
                }
                for task_id, description, files, deps in tasks
            ],
            "rationale": rationale,
        }
    )


def envelope(
    summary: str,
    files: list[dict[str, str]],
    *,
    complete: bool = True,
    issues: list[str] | None = None,
) -> str:
    """A canned worker edit envelope: one JSON object, as the prompt asks for."""
    return json.dumps(
        {"summary": summary, "complete": complete, "issues": issues or [], "files": files}
    )


def write_entry(path: str, content: str) -> dict[str, str]:
    return {"path": path, "action": "write", "content": content}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An empty, self-contained project root. Nothing here escapes ``tmp_path``."""
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    return root


@pytest.fixture
def scripted_tester(monkeypatch: pytest.MonkeyPatch):
    """Replace ``TesterAgent.run`` with a replay queue of verdicts.

    Scripted at the exact seam the orchestrator calls, so the loop, the
    scheduler and the worker are all real. The last verdict repeats forever,
    which is what makes "keep failing" and "fail then pass" both expressible.
    """
    queue: list[TestResult] = []

    def install(*verdicts: TestResult) -> None:
        queue[:] = list(verdicts)

    async def fake_run(
        self: tester_module.TesterAgent, goal: str, tasks: list[TaskModel]
    ) -> TestResult:
        assert goal == GOAL, "the orchestrator handed the tester a rewritten goal"
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        return item

    monkeypatch.setattr(tester_module.TesterAgent, "run", fake_run)
    return install


def pass_verdict(summary: str = "stub tester: green") -> TestResult:
    return TestResult(passed=True, summary=summary, failures=[])


def fail_verdict(*failures: str) -> TestResult:
    return TestResult(
        passed=False,
        summary=f"stub tester: {len(failures)} failure(s)",
        failures=list(failures),
    )


# --------------------------------------------------------------------------- #
# The loop itself
# --------------------------------------------------------------------------- #


async def test_loop_runs_several_cycles_and_finishes_with_the_goal_met(
    workspace: Path, scripted_tester
) -> None:
    """Plan -> work -> verify, twice, with the second cycle's bytes on disk.

    This is the regression that used to end the loop after a single cycle: the
    cycle-2 writes are only reachable if ``Orchestrator._get_tasks()`` reports the
    board (rather than an empty list) and ``_is_done()`` stays false until the
    tester passes.
    """
    cycle_one_plan = plan_json(
        ("t1", "create the health module", ["src/health.py"], []),
        ("t2", "wire the health route", ["src/routes.py"], ["t1"]),
    )
    # Cycle 2 writes different bytes to the same two paths, so the final content on
    # disk is the proof that a second pass really happened after the first tester
    # failure steered it.
    replies = [
        envelope("added health v1", [write_entry("src/health.py", "HEALTH_V1\n")]),
        envelope("added route v1", [write_entry("src/routes.py", "ROUTES_V1\n")]),
        envelope("hardened health", [write_entry("src/health.py", "HEALTH_V2\n")]),
        envelope("hardened route", [write_entry("src/routes.py", "ROUTES_V2\n")]),
    ]
    settings = make_settings(workspace, max_cycles=2)
    client = FakeClient(settings, [cycle_one_plan], replies)
    scripted_tester(
        fail_verdict("tests/test_health.py::test_health 503", "ruff: unused import"),
        pass_verdict("stub tester: green"),
    )

    events: list[str] = []
    orchestrator = Orchestrator(goal=GOAL, client=client, on_event=events.append, settings=settings)
    await orchestrator.run()

    assert orchestrator.cycles == 2, f"the loop stopped early: {orchestrator.stop_reason!r}"
    assert orchestrator.is_done is True
    assert orchestrator.stop_reason == "goal complete"
    assert (workspace / "src" / "health.py").read_text(encoding="utf-8") == "HEALTH_V2\n"
    assert (workspace / "src" / "routes.py").read_text(encoding="utf-8") == "ROUTES_V2\n"

    # Both tasks ran in both cycles, and the plan's dependency edge really was
    # honoured: the dependent worker saw its dependency's result in its prompt.
    worker_prompts = client.prompts_for(WORKER_MODEL)
    assert len(worker_prompts) == 4, "expected two worker runs per cycle"
    assert "## Completed dependencies" in worker_prompts[1]
    assert "[task t1] added health v1" in worker_prompts[1]

    assert all(task.status == "completed" for task in orchestrator.tasks)
    assert [task.id for task in orchestrator.tasks] == ["t1", "t2"]
    assert "Goal complete:" in events[-1]


async def test_failing_verification_keeps_the_original_goal_and_stops_with_a_reason(
    workspace: Path, scripted_tester
) -> None:
    """A permanently red suite must not rewrite the goal, and must not spin forever.

    ``max_cycles=0`` means "unlimited", so the only thing that can end this run is
    the no-progress guard: after ``MAX_IDENTICAL_FAILURES`` cycles with the same
    failure set the loop stops and says so.
    """
    settings = make_settings(workspace, max_cycles=0)
    client = FakeClient(
        settings,
        [plan_json(("t1", "add the health check", ["src/health.py"], []))],
        [envelope("wrote the module", [write_entry("src/health.py", "ATTEMPT\n")])],
    )
    scripted_tester(fail_verdict("tests/test_health.py::test_health FAILED"))

    orchestrator = Orchestrator(goal=GOAL, client=client, settings=settings)
    await orchestrator.run()

    assert orchestrator.goal == GOAL, "the original objective must survive refinement"
    assert orchestrator.focus != GOAL
    assert "tests/test_health.py::test_health FAILED" in orchestrator.focus
    assert orchestrator.is_done is False
    assert orchestrator.cycles == MAX_IDENTICAL_FAILURES, "the loop stopped at the wrong cycle"
    assert orchestrator.stop_reason == (
        f"no progress after {MAX_IDENTICAL_FAILURES} identical failures"
    )


async def test_cycle_budget_stops_the_loop_with_a_budget_reason(
    workspace: Path, scripted_tester
) -> None:
    """``max_cycles`` is a hard stop, reported as such rather than as a failure."""
    settings = make_settings(workspace, max_cycles=1)
    client = FakeClient(
        settings,
        [plan_json(("t1", "add the health check", ["src/health.py"], []))],
        [envelope("wrote the module", [write_entry("src/health.py", "ATTEMPT\n")])],
    )
    scripted_tester(fail_verdict("still red"))

    orchestrator = Orchestrator(goal=GOAL, client=client, settings=settings)
    await orchestrator.run()

    assert orchestrator.cycles == 1
    assert orchestrator.is_done is False
    assert orchestrator.stop_reason == "max cycles reached (1)"


async def test_on_event_emits_single_lines_and_the_final_stop_reason(
    workspace: Path, scripted_tester
) -> None:
    """The event stream is what the web UI renders, so it must stay one line each."""
    settings = make_settings(workspace, max_cycles=2)
    client = FakeClient(
        settings,
        [plan_json(("t1", "add the health check", ["src/health.py"], []))],
        [envelope("wrote the module", [write_entry("src/health.py", "ATTEMPT\n")])],
    )
    scripted_tester(
        fail_verdict("still red\n\npytest summary line 1\npytest summary line 2")
    )

    events: list[str] = []
    orchestrator = Orchestrator(goal=GOAL, client=client, on_event=events.append, settings=settings)
    await orchestrator.run()

    assert events, "the loop reported nothing at all"
    for line in events:
        assert isinstance(line, str)
        assert "\n" not in line and "\r" not in line, f"multi-line event: {line!r}"
        assert line == line.strip() and line, f"padded or empty event: {line!r}"
    # One cycle's worth of events: planning, the plan, running, tasks, testing, stopping.
    assert "Cycle 1: planning" in events
    assert "Cycle 1: plan ready with 1 task(s)" in events
    assert any(line.startswith("Cycle 1: tests failed") for line in events)
    assert f"Stopping: {orchestrator.stop_reason}" in events
    assert events[-1] == f"Stopping: {orchestrator.stop_reason}"


# --------------------------------------------------------------------------- #
# The write path, through the real WorkerAgent
# --------------------------------------------------------------------------- #


def make_worker(
    workspace: Path, reply: str, task_id: str = "t1", **setting_overrides: Any
) -> WorkerAgent:
    settings = make_settings(workspace, **setting_overrides)
    client = FakeClient(settings, [], [reply])
    task = TaskModel(id=task_id, description="implement the health check", files=["src/health.py"])
    return WorkerAgent(task=task, client=client, settings=settings)


async def test_worker_actually_modifies_files_on_disk(workspace: Path) -> None:
    """The headline claim: a worker reply becomes real bytes in the workspace."""
    reply = envelope(
        "implemented the health check",
        [write_entry("src/health.py", "def health() -> str:\n    return 'ok'\n")],
    )
    worker = make_worker(workspace, reply)

    summary = await worker.run()

    target = workspace / "src" / "health.py"
    assert target.read_text(encoding="utf-8") == "def health() -> str:\n    return 'ok'\n"
    assert worker.advisory is False
    assert [(record.action, record.path) for record in worker.applied] == [
        (ACTION_CREATED, "src/health.py")
    ]
    assert "created src/health.py" in summary
    assert "implemented the health check" in summary
    assert "complete: yes" in summary.lower()


async def test_overwrite_is_backed_up_and_the_backup_is_restorable(workspace: Path) -> None:
    """Overwriting a file keeps the original bytes somewhere it can be put back."""
    target = workspace / "src" / "health.py"
    original = "ORIGINAL = 'keep me'\n"
    target.write_text(original, encoding="utf-8")

    reply = envelope("replaced the module", [write_entry("src/health.py", "REPLACEMENT = 1\n")])
    worker = make_worker(workspace, reply)

    summary = await worker.run()

    assert target.read_text(encoding="utf-8") == "REPLACEMENT = 1\n"
    assert [record.action for record in worker.applied] == [ACTION_MODIFIED]
    assert worker.applied[0].backup_path, "an overwrite must record where the original went"

    backups = sorted((workspace / ".furrow" / "backups").rglob("src/health.py"))
    assert len(backups) == 1, f"expected exactly one backup, found {backups}"
    backup = backups[0]
    # The stamp is the current UTC time, so assert the layout rather than the value.
    assert backup.parents[1].name.endswith("-t1"), backup.parents[1].name
    assert backup.read_text(encoding="utf-8") == original
    assert str(backup) in summary

    # "Restorable" means the copy is the real thing: put it back and the original
    # is what is on disk again.
    target.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    assert target.read_text(encoding="utf-8") == original


async def test_hostile_edits_are_rejected_reported_and_never_written(
    workspace: Path, tmp_path: Path
) -> None:
    """Traversal, protected trees and symlinked directories all fail closed.

    A rejected edit must show up in the summary *and* in ``worker.applied``; a
    refusal the next planning cycle cannot see is a refusal it will re-request.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("TOPSECRET\n", encoding="utf-8")
    (workspace / "link").symlink_to(outside, target_is_directory=True)

    git_config = workspace / ".git" / "config"
    git_config.parent.mkdir()
    git_config.write_text("[core]\n\trepositoryformatversion = 0\n", encoding="utf-8")

    reply = envelope(
        "tried three things, one worked",
        [
            write_entry("src/health.py", "REAL = 1\n"),
            write_entry("../escape.py", "PWNED\n"),
            write_entry(".git/config", "[core]\n\tbare = true\n"),
            write_entry("link/secret.txt", "PWNED\n"),
            write_entry("/etc/furrow-pwned", "PWNED\n"),
            write_entry("module.pyc", b"\x00\x01".decode("latin-1")),
        ],
    )
    worker = make_worker(workspace, reply)

    summary = await worker.run()

    # Nothing hostile reached the disk.
    assert not (tmp_path / "escape.py").exists()
    assert not Path("/etc/furrow-pwned").exists()
    assert secret.read_text(encoding="utf-8") == "TOPSECRET\n"
    assert git_config.read_text(encoding="utf-8") == "[core]\n\trepositoryformatversion = 0\n"
    assert not (workspace / "module.pyc").exists()
    # The legitimate edit in the same reply still landed: rejection is per entry.
    assert (workspace / "src" / "health.py").read_text(encoding="utf-8") == "REAL = 1\n"

    skipped = [record for record in worker.applied if record.action == ACTION_SKIPPED]
    assert len(skipped) == 5, [record.reason for record in worker.applied]
    assert len(worker.applied) == 6, "every requested edit must produce exactly one record"
    reasons = " | ".join(record.reason for record in skipped)
    assert "escapes the workspace with '..'" in reasons
    assert "writes into .git/ are not allowed" in reasons
    assert "resolves outside the workspace" in reasons
    assert "must be relative to the workspace" in reasons
    assert "generated bytecode" in reasons
    # The rejections are visible to the next planning cycle, not silently dropped.
    assert "Rejected or skipped 5 requested edit(s)" in summary
    for label in ("../escape.py", ".git/config", "link/secret.txt", "/etc/furrow-pwned"):
        assert label in summary


async def test_apply_changes_false_leaves_the_workspace_untouched(workspace: Path) -> None:
    """Advisory mode reports what it would have written and writes nothing."""
    target = workspace / "src" / "health.py"
    target.write_text("ORIGINAL\n", encoding="utf-8")

    reply = envelope(
        "here is what I would change",
        [
            write_entry("src/health.py", "REPLACEMENT\n"),
            write_entry("src/brand_new.py", "NEW\n"),
        ],
    )
    worker = make_worker(workspace, reply, apply_changes=False)

    summary = await worker.run()

    assert target.read_text(encoding="utf-8") == "ORIGINAL\n"
    assert not (workspace / "src" / "brand_new.py").exists()
    assert not (workspace / ".furrow").exists(), "advisory mode must not even create backups"
    assert worker.applied and all(record.action == ACTION_SKIPPED for record in worker.applied)
    assert "apply_changes is disabled" in summary
    # Still reported, so the caller can see what the model proposed.
    assert "src/brand_new.py" in summary


async def test_a_worker_cannot_overwrite_its_own_backups(workspace: Path) -> None:
    """The backup tree is the recovery path; a model reply must not be able to clobber it.

    ``atomic_write_text`` ends in ``os.replace``, which overwrites unconditionally,
    and ``.furrow`` is not a forbidden directory - so without an explicit guard the
    safety net sits inside the blast radius of the thing it protects.
    """
    target = workspace / "src" / "health.py"
    original = "ORIGINAL = 'keep me'\n"
    target.write_text(original, encoding="utf-8")

    first = make_worker(workspace, envelope("first take", [write_entry("src/health.py", "V1\n")]))
    await first.run()
    backups = sorted((workspace / ".furrow" / "backups").rglob("src/health.py"))
    assert len(backups) == 1
    backup = backups[0]
    assert backup.read_text(encoding="utf-8") == original

    sabotage_path = backup.relative_to(workspace).as_posix()
    second = make_worker(workspace, envelope("oops", [write_entry(sabotage_path, "DESTROYED\n")]))
    summary = await second.run()

    assert backup.read_text(encoding="utf-8") == original, "a worker reply destroyed a backup"
    assert all(record.action == ACTION_SKIPPED for record in second.applied)
    assert "backup" in summary and "not allowed" in summary


async def test_one_bad_file_does_not_abort_the_rest_of_the_reply(workspace: Path) -> None:
    """A single unencodable payload must be one rejected edit, not a lost envelope.

    ``json.loads`` happily turns ``"\\ud800"`` into a lone surrogate, which cannot be
    encoded as UTF-8. Letting that escape ``apply_envelope_files`` would discard the
    records for files that were already written - an edit on disk that nothing in the
    product knows about - and abort the remaining edits in the same reply.
    """
    reply = envelope(
        "three files, one unencodable",
        [
            write_entry("src/first.py", "FIRST = 1\n"),
            {"path": "src/bad.py", "action": "write", "content": "BAD = \ud800\n"},
            write_entry("src/last.py", "LAST = 2\n"),
        ],
    )
    worker = make_worker(workspace, reply)

    summary = await worker.run()

    # The run completed and reported, rather than raising.
    assert isinstance(summary, str) and summary
    assert (workspace / "src" / "first.py").read_text(encoding="utf-8") == "FIRST = 1\n"
    assert (workspace / "src" / "last.py").read_text(encoding="utf-8") == "LAST = 2\n"
    assert not (workspace / "src" / "bad.py").exists()

    actions = [(record.action, record.path) for record in worker.applied]
    assert actions == [
        (ACTION_CREATED, "src/first.py"),
        (ACTION_SKIPPED, "src/bad.py"),
        (ACTION_CREATED, "src/last.py"),
    ], "every requested edit must produce exactly one record, applied edits included"
    assert "UTF-8" in " | ".join(record.reason for record in worker.applied)
    # No temp file leaked into the workspace.
    assert not [p for p in (workspace / "src").iterdir() if p.name.startswith(".furrow-tmp-")]


async def test_repeated_overwrites_of_one_path_keep_every_generation(workspace: Path) -> None:
    """Two writes to the same path in one reply must not destroy the first backup.

    ``create_backup`` refuses to clobber, so the second copy lands beside the first
    under a numeric suffix. Without that, the only copy of the original bytes would
    be overwritten by the intermediate content.
    """
    target = workspace / "src" / "health.py"
    original = "ORIGINAL = 0\n"
    target.write_text(original, encoding="utf-8")

    reply = envelope(
        "wrote the same path twice",
        [
            write_entry("src/health.py", "GENERATION = 1\n"),
            write_entry("src/health.py", "GENERATION = 2\n"),
        ],
    )
    worker = make_worker(workspace, reply)
    await worker.run()

    assert target.read_text(encoding="utf-8") == "GENERATION = 2\n"
    backups = sorted(p for p in (workspace / ".furrow" / "backups").rglob("*") if p.is_file())
    assert len(backups) == 2, f"expected two generations to be kept, found {backups}"
    assert sorted(p.read_text(encoding="utf-8") for p in backups) == sorted(
        [original, "GENERATION = 1\n"]
    )


async def test_write_caps_bound_a_single_reply(workspace: Path) -> None:
    """Three independent caps stop one reply from rewriting the whole repository."""
    reply = envelope(
        "wrote as much as I could",
        [write_entry(f"src/mod_{index}.py", "X = 1\n") for index in range(4)]
        + [write_entry("src/huge.py", "y" * 500)],
    )
    worker = make_worker(
        workspace,
        reply,
        max_files_per_task=2,
        max_write_bytes=100,
        max_total_write_bytes=150,
    )

    summary = await worker.run()

    assert (workspace / "src" / "mod_0.py").exists()
    assert (workspace / "src" / "mod_1.py").exists()
    assert not (workspace / "src" / "mod_2.py").exists(), "max_files_per_task was not enforced"
    assert not (workspace / "src" / "huge.py").exists(), "max_write_bytes was not enforced"

    reasons = " | ".join(record.reason for record in worker.applied)
    assert "max_files_per_task" in reasons
    assert "max_write_bytes" in reasons
    # Rejections are reported, not swallowed.
    assert "max_files_per_task" in summary and "max_write_bytes" in summary
    assert [record.action for record in worker.applied[:2]] == [ACTION_CREATED, ACTION_CREATED]
    assert all(record.action == ACTION_SKIPPED for record in worker.applied[2:])


async def test_total_write_bytes_cap_bounds_one_reply(workspace: Path) -> None:
    """The total cap bites even when every individual file is under the per-file one."""
    reply = envelope(
        "wrote two files",
        [
            write_entry("src/a.py", "A" * 80),
            write_entry("src/b.py", "B" * 80),
        ],
    )
    worker = make_worker(
        workspace, reply, max_files_per_task=10, max_write_bytes=200, max_total_write_bytes=100
    )

    summary = await worker.run()

    assert (workspace / "src" / "a.py").read_text(encoding="utf-8") == "A" * 80
    assert not (workspace / "src" / "b.py").exists(), "max_total_write_bytes was not enforced"
    assert "max_total_write_bytes" in summary


async def test_prose_only_reply_writes_nothing_and_is_flagged_advisory(workspace: Path) -> None:
    """A worker that decides to change nothing is a supported outcome, not a failure."""
    worker = make_worker(
        workspace,
        "I looked at src/health.py and the goal is already satisfied by it, "
        "so I did not change anything.",
    )

    summary = await worker.run()

    assert worker.advisory is True
    assert not (workspace / "src" / "health.py").exists()
    assert summary == (
        "I looked at src/health.py and the goal is already satisfied by it, "
        "so I did not change anything."
    )


# --------------------------------------------------------------------------- #
# The tester half of a cycle, real agent with the command layer stubbed
# --------------------------------------------------------------------------- #


async def test_real_tester_treats_a_missing_runner_as_a_vacuous_pass(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No runner detected must be a green cycle, not a permanent fake failure.

    ``TesterAgent.run`` is real here; only runner detection is stubbed, so the
    vacuous-pass branch is genuinely exercised and no subprocess is spawned.
    """
    monkeypatch.setattr(tester_module, "detect_test_commands", lambda **_: [])
    settings = make_settings(workspace)
    client = FakeClient(settings, [], [])

    result = await tester_module.TesterAgent(client=client, settings=settings).run(
        GOAL, [TaskModel(id="t1", description="do the thing")]
    )

    assert result.passed is True
    assert result.failures == []
    assert tester_module.NO_RUNNER_SUMMARY in result.summary
    assert client.prompts == [], "a vacuous pass must not spend an LLM call"


async def test_real_tester_will_not_let_a_passing_llm_verdict_override_a_failed_run(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit codes are the evidence; a model claiming "passed" cannot overturn one.

    Detection and the command itself are stubbed to a single failing ``pytest``,
    so the assertion is about the exit-code verdict path and nothing else.
    """
    argv = ["pytest", "-q"]
    monkeypatch.setattr(tester_module, "detect_test_commands", lambda **_: [argv])
    # Lint/type checks are out of scope here: the point is the test-runner exit code.
    monkeypatch.setattr(tester_module, "detect_static_checks", lambda **_: [])

    async def failing_command(
        run_argv: list[str], *, cwd: Any = None, env: Any = None, timeout: Any = None
    ) -> tester_module.CommandOutcome:
        assert run_argv == argv
        return tester_module.CommandOutcome(
            argv=argv,
            label="pytest",
            exit_code=1,
            stdout="1 failed",
            stderr="",
        )

    monkeypatch.setattr(tester_module, "run_command", failing_command)
    settings = make_settings(workspace)
    client = FakeClient(
        settings, [], [], [json.dumps({"passed": True, "summary": "all good"})]
    )

    result = await tester_module.TesterAgent(client=client, settings=settings).run(GOAL, [])

    assert result.passed is False, "an LLM 'passed: true' overruled a non-zero exit code"
    assert any("pytest" in failure for failure in result.failures)
    assert "[overridden-by-exit-codes]" in result.summary


# --------------------------------------------------------------------------- #
# The installed entry point, in a real subprocess
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]

# Options the README and --help both document for ``start``. If a flag is renamed
# or dropped, this is what notices.
START_OPTIONS = (
    "--provider",
    "--model",
    "--planner-model",
    "--worker-model",
    "--tester-model",
    "--max-cycles",
    "--max-parallel-tasks",
    "--workspace",
    "--log-level",
)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI the way an installed console script would, in a subprocess."""
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "furrow.cli.main", *args],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO_ROOT,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", "/tmp"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(REPO_ROOT),
        },
    )


def test_cli_help_and_version_work_without_a_key() -> None:
    """--version, start --help and web --help must all exit 0 and print real text."""
    # Run concurrently: importing the CLI pulls in both LLM SDKs, so each fresh
    # interpreter costs ~2s of import time before click does anything.
    with ThreadPoolExecutor(max_workers=3) as pool:
        version, start, web = list(
            pool.map(
                lambda argv: run_cli(*argv),
                (("--version",), ("start", "--help"), ("web", "--help")),
            )
        )

    assert version.returncode == 0, version.stderr
    assert version.stdout.strip().startswith("furrow "), version.stdout

    assert start.returncode == 0, start.stderr
    assert "Run the development loop" in start.stdout
    for option in START_OPTIONS:
        assert option in start.stdout, f"{option} is missing from `start --help`"

    assert web.returncode == 0, web.stderr
    assert "--host" in web.stdout and "--port" in web.stdout
    # show_default renders the value itself, wrapped across lines; an unsubstituted
    # placeholder would be printed verbatim to the user, which is not help text.
    collapsed = " ".join(web.stdout.split())
    assert "%(default)s" not in collapsed and "%default" not in collapsed
    assert "[default: 127.0.0.1]" in collapsed
    assert "[default: 8000]" in collapsed
    assert "loopback" in collapsed.lower(), "the host help must still state the loopback default"