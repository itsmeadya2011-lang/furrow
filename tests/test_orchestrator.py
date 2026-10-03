"""The plan -> execute -> test loop, its termination rules, and its event hook."""

from __future__ import annotations

import io
import json
import shlex
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from furrow import config
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import Orchestrator
from tests.conftest import PLANNER_MARKER, TESTER_MARKER, WORKER_MARKER, FakeLLM, make_settings

GOAL = "ship the thing"

GREEN = '{"passed": true, "summary": "green", "failures": []}'
RED = '{"passed": false, "summary": "1 failed", "failures": ["tests/test_x.py::test_y"]}'
PASSING_COMMAND = f"{shlex.quote(sys.executable)} -c {shlex.quote('raise SystemExit(0)')}"
FAILING_COMMAND = f"{shlex.quote(sys.executable)} -c {shlex.quote('raise SystemExit(1)')}"


def task(tid: str, description: str, dependencies: list[str] | None = None, **extra) -> dict:
    return {
        "id": tid,
        "description": description,
        "files": [],
        "dependencies": dependencies or [],
        **extra,
    }


def plan(*tasks: dict, complete: bool = False, rationale: str = "because") -> str:
    return json.dumps({"tasks": list(tasks), "rationale": rationale, "complete": complete})


def cycling(*payloads: object):
    """A route value that hands out each payload in turn, repeating the last."""
    queue = list(payloads)
    return lambda prompt: queue.pop(0) if len(queue) > 1 else queue[0]


def orchestrator_fake(
    tmp_workspace: Path,
    *,
    plan_payload: object,
    worker: object = "did the work",
    tester: str = GREEN,
    extra_routes: dict | None = None,
    **settings_overrides,
) -> FakeLLM:
    """A scripted client wired up like the orchestrator expects.

    ``max_cycles`` defaults to 2 so that a planner which keeps finding work
    cannot spin forever; pass a higher cap to prove a *different* stop condition
    is what ended the run.
    """
    routes = dict(extra_routes or {})
    routes.update({PLANNER_MARKER: plan_payload, WORKER_MARKER: worker, TESTER_MARKER: tester})
    options: dict = {
        "workspace": tmp_workspace,
        "test_command": PASSING_COMMAND,
        "test_timeout": 30,
        "max_cycles": 2,
    }
    options.update(settings_overrides)
    return FakeLLM(routes=routes, settings=make_settings(**options))


@pytest.fixture(autouse=True)
def console_sink() -> Iterator[io.StringIO]:
    """Keep the orchestrator's rich output out of the test log."""
    sink = io.StringIO()
    original = orchestrator_module.console
    orchestrator_module.console = Console(file=sink, width=100, no_color=True)
    yield sink
    orchestrator_module.console = original


class Recorder:
    """An ``on_event`` sink that records everything it is handed."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def __call__(self, message: str) -> None:
        self.events.append(message)

    def joined(self) -> str:
        return "\n".join(str(event) for event in self.events)


def statuses(orch: Orchestrator) -> dict[str, config.TaskStatus]:
    return {item.id: item.status for item in orch._get_tasks()}


# --------------------------------------------------------- termination guards


def test_a_fresh_orchestrator_is_not_done(tmp_workspace: Path) -> None:
    # The one-cycle regression: ``_get_tasks()`` used to return ``[]``, so the very
    # first ``_is_done()`` check was already True and the "infinite" loop always
    # halted after a single cycle.
    fake = orchestrator_fake(tmp_workspace, plan_payload=plan(task("1", "build A")))
    orch = Orchestrator(GOAL, client=fake)

    assert orch.cycles == 0
    assert orch._is_done() is False
    assert orch.plan is None
    assert orch.goal == GOAL
    assert orch.current_focus == GOAL
    assert fake.calls == []  # nothing has happened yet


async def test_get_tasks_tracks_real_state_after_a_cycle(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(tmp_workspace, plan_payload=plan(task("1", "build A")))
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert orch._get_tasks() == orch.tasks
    assert [item.id for item in orch._get_tasks()] == ["1"]
    assert orch.tasks[0].status is config.TaskStatus.COMPLETED


async def test_max_cycles_stops_the_loop_after_exactly_three_cycles(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=plan(task("1", "build A"), task("2", "build B")),
        max_cycles=3,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert orch.cycles == 3
    assert len(fake.calls_matching(PLANNER_MARKER)) == 3
    assert len(fake.calls_matching(WORKER_MARKER)) == 6  # two tasks per cycle


async def test_an_empty_task_list_halts_immediately(tmp_workspace: Path, console_sink) -> None:
    fake = orchestrator_fake(tmp_workspace, plan_payload=plan(), max_cycles=5)
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert orch.cycles == 1
    assert orch._last_plan_had_no_tasks is True
    assert fake.calls_matching(WORKER_MARKER) == []
    assert "No tasks planned" in console_sink.getvalue()


async def test_plan_complete_halts_after_one_green_cycle(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(
        tmp_workspace, plan_payload=plan(task("1", "build A"), complete=True), max_cycles=5
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert orch.cycles == 1
    assert orch._last_test_passed is True
    assert orch.plan is not None and orch.plan.complete is True


async def test_plan_complete_does_not_halt_while_the_suite_is_red(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=plan(task("1", "build A"), complete=True),
        tester=RED,
        test_command=FAILING_COMMAND,
        max_cycles=5,
        max_idle_cycles=2,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert orch._last_test_passed is False
    assert orch.cycles == 2  # stopped by max_idle_cycles, not by the plan
    assert orch._idle_cycles == 2


async def test_max_idle_cycles_stops_a_loop_that_makes_no_progress(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=plan(task("1", "already done", status="completed")),
        test_command=FAILING_COMMAND,
        max_cycles=5,
        max_idle_cycles=2,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    # Every cycle re-plans the same finished task, so nothing is ever executed.
    assert fake.calls_matching(WORKER_MARKER) == []
    assert orch.cycles == 2
    assert orch._idle_cycles == 2


async def test_a_missing_test_runner_is_not_treated_as_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path, console_sink
) -> None:
    monkeypatch.setenv("PATH", "")
    fake = orchestrator_fake(
        tmp_workspace, plan_payload=plan(task("1", "build A"), complete=True), test_command=None
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert orch.cycles == 1
    assert orch.tasks[0].status is config.TaskStatus.COMPLETED
    assert "No test runner" in console_sink.getvalue()
    assert "Tests failed" not in console_sink.getvalue()


async def test_a_planner_failure_does_not_kill_the_loop(tmp_workspace: Path, console_sink) -> None:
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload="I would rather write an essay.",
        max_cycles=5,
        max_idle_cycles=2,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert orch.cycles == 2
    assert orch._idle_cycles == 2
    assert fake.calls_matching(WORKER_MARKER) == []
    assert "Planning failed" in console_sink.getvalue()


async def test_the_loop_survives_an_unexpected_cycle_error(
    monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path, console_sink
) -> None:
    original = Orchestrator._execute_tasks
    seen: list[int] = []

    async def flaky(self, tasks):
        seen.append(1)
        if len(seen) == 1:
            raise RuntimeError("kaboom")
        return await original(self, tasks)

    monkeypatch.setattr(Orchestrator, "_execute_tasks", flaky)
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=cycling(plan(task("A", "build a")), plan(task("A", "build a"), complete=True)),
        max_cycles=3,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert orch.cycles == 2
    assert len(seen) == 2
    assert "aborted" in console_sink.getvalue()
    assert orch.tasks[0].status is config.TaskStatus.COMPLETED


# ------------------------------------------------------------------- goal drift


async def test_goal_is_never_reassigned_by_a_failing_test(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=plan(task("1", "build A")),
        tester=RED,
        test_command=FAILING_COMMAND,
        max_cycles=2,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert orch.goal == GOAL
    assert GOAL in orch.current_focus
    assert "tests/test_x.py::test_y" in orch.current_focus
    assert "ALREADY COMPLETED" in orch.current_focus
    # Every planner prompt still leads with the original objective.
    for prompt, _ in fake.calls_matching(PLANNER_MARKER):
        assert f"ORIGINAL GOAL (this never changes):\n{GOAL}" in prompt


def test_focus_grows_with_completed_work_and_failures(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(tmp_workspace, plan_payload=plan(task("1", "build A")))
    orch = Orchestrator(GOAL, client=fake)

    focus = orch._build_focus(None)
    assert GOAL in focus
    assert "ALREADY COMPLETED" not in focus

    orch.completed_summaries.append("[1] did the work")
    failing = config.TestResult(passed=False, summary="boom", failures=["assert 1 == 2"])
    focus = orch._build_focus(failing)
    assert "did the work" in focus
    assert "Fix these failing tests" in focus
    assert "assert 1 == 2" in focus


# ------------------------------------------------------------- task scheduling


async def test_dependencies_are_executed_in_order(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=plan(task("A", "build the base"), task("B", "build the rest", ["A"])),
        extra_routes={"Task: build the base": "base done", "Task: build the rest": "rest done"},
        max_cycles=1,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert orch.completed_summaries == ["[A] base done", "[B] rest done"]
    worker_prompts = [prompt for prompt, _ in fake.calls_matching(WORKER_MARKER)]
    assert len(worker_prompts) == 2
    assert "Task: build the base" in worker_prompts[0]
    assert "Task: build the rest" in worker_prompts[1]
    assert "Depends on these completed tasks: A" in worker_prompts[1]
    assert statuses(orch) == {"A": config.TaskStatus.COMPLETED, "B": config.TaskStatus.COMPLETED}


async def test_independent_tasks_run_together(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=plan(task("A", "build a"), task("B", "build b")),
        max_parallel_tasks=2,
        max_cycles=1,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert statuses(orch) == {"A": config.TaskStatus.COMPLETED, "B": config.TaskStatus.COMPLETED}


async def test_max_parallel_tasks_caps_observed_concurrency(tmp_workspace: Path) -> None:
    tasks = [task(str(index), f"build number {index}") for index in range(6)]
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=plan(*tasks),
        max_parallel_tasks=2,
        max_cycles=1,
    )
    fake.delay = 0.02
    fake.track_concurrency = True
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert len(fake.calls_matching(WORKER_MARKER)) == 6
    assert fake.max_active == 2  # both slots used, never exceeded
    assert all(item.status is config.TaskStatus.COMPLETED for item in orch.tasks)


async def test_a_dependency_cycle_blocks_its_tasks_instead_of_hanging(
    tmp_workspace: Path, console_sink
) -> None:
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=plan(task("A", "build a", ["B"]), task("B", "build b", ["A"])),
        max_idle_cycles=1,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert statuses(orch) == {"A": config.TaskStatus.BLOCKED, "B": config.TaskStatus.BLOCKED}
    assert fake.calls_matching(WORKER_MARKER) == []
    assert all("cycle" in (item.result or "") for item in orch.tasks)
    assert "Dependency cycle detected" in console_sink.getvalue()


async def test_a_failed_dependency_blocks_its_dependents(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=plan(
            task("A", "build the base"),
            task("B", "build the rest", ["A"]),
            task("C", "build something else"),
        ),
        extra_routes={"Task: build the base": RuntimeError("worker exploded")},
        max_idle_cycles=1,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert statuses(orch) == {
        "A": config.TaskStatus.FAILED,
        "B": config.TaskStatus.BLOCKED,
        "C": config.TaskStatus.COMPLETED,
    }
    assert orch.tasks[0].result is not None and "worker exploded" in orch.tasks[0].result
    assert orch.tasks[1].result == "A dependency failed."


async def test_a_failed_worker_marks_the_task_and_the_run_continues(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=plan(task("A", "build a"), task("B", "build b")),
        extra_routes={"Task: build a": RuntimeError("no such module")},
        max_cycles=1,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert statuses(orch) == {"A": config.TaskStatus.FAILED, "B": config.TaskStatus.COMPLETED}
    assert orch.cycles == 1  # a failed task does not abort the run
    assert orch._last_test_result is not None and orch._last_test_result.passed is True


async def test_a_dependency_on_an_unknown_task_is_treated_as_runnable(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=plan(task("A", "build a", ["nonexistent"])),
        max_cycles=1,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert statuses(orch) == {"A": config.TaskStatus.COMPLETED}


async def test_already_completed_tasks_are_not_reexecuted(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=plan(task("A", "build a", status="completed"), task("B", "build b")),
        max_cycles=1,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    worker_prompts = [prompt for prompt, _ in fake.calls_matching(WORKER_MARKER)]
    assert len(worker_prompts) == 1
    assert "Task: build b" in worker_prompts[0]
    assert orch.completed_summaries == ["[B] did the work"]


async def test_completed_summaries_are_capped(tmp_workspace: Path) -> None:
    first = plan(*[task(str(index), f"build {index}") for index in range(12)])
    second = plan(*[task(str(index), f"build {index}") for index in range(12, 24)])
    fake = orchestrator_fake(
        tmp_workspace,
        plan_payload=cycling(first, second),
        max_parallel_tasks=12,
        max_cycles=2,
    )
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert len(orch.completed_summaries) == 20
    assert orch.completed_summaries[-1].startswith("[23]")


# ----------------------------------------------------------------- event hook


async def test_on_event_receives_plain_strings_only(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(
        tmp_workspace, plan_payload=plan(task("A", "build a"), complete=True), max_cycles=1
    )
    recorder = Recorder()
    orch = Orchestrator(GOAL, client=fake, on_event=recorder)
    await orch.run()

    assert recorder.events, "the hook must be called"
    assert all(isinstance(event, str) for event in recorder.events)
    assert not any(isinstance(event, Panel | Pretty) for event in recorder.events)
    text = recorder.joined()
    assert "Furrow" in text  # the header panel rendered to text
    assert "Summary" in text and "Cycles:" in text  # the final report panel too
    assert "Cycle 1" in text
    assert "Tests passed" in text


async def test_a_broken_event_callback_does_not_stop_the_loop(tmp_workspace: Path) -> None:
    def explode(message: str) -> None:
        raise RuntimeError("consumer is gone")

    fake = orchestrator_fake(
        tmp_workspace, plan_payload=plan(task("A", "build a"), complete=True), max_cycles=3
    )
    orch = Orchestrator(GOAL, client=fake, on_event=explode)
    await orch.run()

    assert orch.cycles == 1
    assert orch.tasks[0].status is config.TaskStatus.COMPLETED


async def test_the_loop_runs_without_any_callback(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(tmp_workspace, plan_payload=plan(task("A", "build a"), complete=True))
    orch = Orchestrator(GOAL, client=fake)
    await orch.run()

    assert orch.cycles == 1


# ------------------------------------------------------------------ reporting


def test_final_report_summarises_the_run(tmp_workspace: Path) -> None:
    fake = orchestrator_fake(tmp_workspace, plan_payload=plan(task("A", "build a")))
    orch = Orchestrator(GOAL, client=fake)
    orch.tasks = [
        config.TaskModel(id="A", description="a", status=config.TaskStatus.COMPLETED),
        config.TaskModel(id="B", description="b", status=config.TaskStatus.FAILED),
    ]
    report = orch._final_report()
    assert isinstance(report, Panel)

    text = orchestrator_module._render_text(report)
    assert "Cycles: 0" in text
    assert "1 completed" in text
    assert "1 failed" in text
    assert "Tests: not run" in text


def test_tests_were_skipped_detects_a_missing_runner() -> None:
    skipped = config.TestResult(passed=False, summary="No test runner found.")
    assert Orchestrator._tests_were_skipped(skipped) is True
    assert (
        Orchestrator._tests_were_skipped(config.TestResult(passed=False, summary="2 failed"))
        is False
    )
    # A green result is never "skipped", even when the summary mentions a runner.
    green = config.TestResult(passed=True, summary="no test runner here")
    assert Orchestrator._tests_were_skipped(green) is False
