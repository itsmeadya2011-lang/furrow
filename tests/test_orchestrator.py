"""Integration tests for the real `Orchestrator`, driven by the scripted client.

Every test here constructs an actual `Orchestrator` (no stubs in the class under test),
gives it a `ScriptedLLMClient` and a `tmp_path` workspace, and then asserts on what the run
produced: the `RunSummary`, the event history, the snapshot and the state file.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, reload_settings
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import Orchestrator, RunSummary
from furrow.events import Event, EventBus, EventType
from furrow.llm import ToolCall, ToolResult
from tests.conftest import DEFAULT_PLAN, DEFAULT_TEST, ScriptedLLMClient, as_llm_client

GOAL = "make the thing work"
RUN_TIMEOUT = 30.0


def command(script: str) -> str:
    """A shell-free command string; the tester re-splits it with shlex, so join it here."""
    return shlex.join([sys.executable, "-c", script])


PASSING_COMMAND = command("import sys; sys.exit(0)")
FAILING_COMMAND = command("import sys; sys.exit(1)")


def build_settings(workspace: Path, **overrides: Any) -> Settings:
    """Test settings with the same bounds as the `settings` fixture, plus overrides."""
    defaults: dict[str, Any] = {
        "workspace": workspace,
        "max_cycles": 2,
        "max_stalled_cycles": 2,
        "test_timeout": 60,
        "state_file": None,
    }
    return reload_settings(**{**defaults, **overrides})


def plan_of(*tasks: dict[str, Any], rationale: str = "scripted") -> dict[str, Any]:
    return {"tasks": list(tasks), "rationale": rationale}


def task_of(task_id: str, depends_on: Sequence[str] = (), **extra: Any) -> dict[str, Any]:
    return {
        "id": task_id,
        "description": f"do {task_id}",
        "files": [f"{task_id}.py"],
        "dependencies": list(depends_on),
        **extra,
    }


def make_orchestrator(
    goal: str,
    client: ScriptedLLMClient,
    settings: Settings,
    bus: EventBus,
) -> Orchestrator:
    return Orchestrator(goal, client=as_llm_client(client), settings=settings, bus=bus)


def events_of(bus: EventBus, kind: EventType) -> list[Event]:
    return [event for event in bus.history() if event.type is kind]


def task_events(bus: EventBus, kind: EventType) -> list[Event]:
    """Only the orchestrator's own task events: the worker tags its events with `task_id`."""
    return [event for event in events_of(bus, kind) if "id" in event.data]


def peak_concurrency(bus: EventBus) -> int:
    """Peak number of simultaneously running tasks, reconstructed from the event stream.

    Measured from the WORKER's own TASK_START/TASK_END pair (tagged `task_id`), because the
    orchestrator emits its TASK_END events in one batch after `asyncio.gather` returns for the
    whole wave: its brackets say nothing about overlap, the worker's do.
    """
    running = 0
    peak = 0
    for event in bus.history():
        if "task_id" not in event.data:
            continue
        if event.type is EventType.TASK_START:
            running += 1
            peak = max(peak, running)
        elif event.type is EventType.TASK_END:
            running -= 1
    return peak


def worker_intervals(bus: EventBus) -> dict[str, tuple[float, float]]:
    """Map task_id -> (start timestamp, end timestamp) from the worker's own event pair."""
    intervals: dict[str, tuple[float, float]] = {}
    for event in bus.history():
        task_id = event.data.get("task_id")
        if task_id is None:
            continue
        started, _ = intervals.get(task_id, (event.timestamp, event.timestamp))
        if event.type is EventType.TASK_START:
            intervals[task_id] = (event.timestamp, started)
        elif event.type is EventType.TASK_END:
            intervals[task_id] = (started, event.timestamp)
    return intervals


async def run_within(orchestrator: Orchestrator, timeout: float = RUN_TIMEOUT) -> RunSummary:
    """Run the orchestrator under a hard timeout so an infinite loop fails, not hangs."""
    return await asyncio.wait_for(orchestrator.run(), timeout=timeout)


class UnusableWorkerClient(ScriptedLLMClient):
    """Planner and tester behave normally; only the worker's tool loop always fails."""

    async def complete_with_tools(
        self,
        prompt: str,
        tools: Sequence[Any],
        *,
        system: str | None = None,
        model: str | None = None,
        max_iterations: int | None = None,
        on_tool_call: Callable[[ToolCall], Awaitable[None]] | None = None,
    ) -> ToolResult:
        self.error_calls += 1
        raise RuntimeError("the worker's model call is broken")


class TesterDownClient(ScriptedLLMClient):
    """The planner answers normally; every tester call raises."""

    __test__ = False  # not a test class, despite the name

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_attempts: int = 2,
    ) -> Any:
        if system == TESTER_PROMPT:
            self.error_calls += 1
            raise RuntimeError("tester is down")
        return await super().complete_json(
            prompt, system=system, model=model, max_attempts=max_attempts
        )


class GatedPlannerClient(ScriptedLLMClient):
    """Blocks the first planner call until the test releases it, so `stop()` can land."""

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        super().__init__(settings, **kwargs)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self._gated = True

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_attempts: int = 2,
    ) -> Any:
        if self._gated:
            self._gated = False
            self.entered.set()
            await self.release.wait()
        return await super().complete_json(
            prompt, system=system, model=model, max_attempts=max_attempts
        )


# ------------------------------------------------------------------ cycle count
async def test_run_executes_exactly_max_cycles_cycles(
    settings: Settings, bus: EventBus
) -> None:
    # Regression: `_get_tasks()` returned a hardcoded [] so `0 >= 0` ended the loop after
    # exactly one cycle no matter what max_cycles said.
    bounded = build_settings(settings.workspace, max_cycles=3, max_stalled_cycles=99)
    client = ScriptedLLMClient(bounded, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, bounded, bus)
    summary = await run_within(orchestrator)
    assert summary.cycles == 3
    assert orchestrator.cycles == 3
    assert summary.halted_reason == "max_cycles"
    starts = events_of(bus, EventType.CYCLE_START)
    assert len(starts) == 3
    assert [event.cycle for event in starts] == [1, 2, 3]
    assert len(events_of(bus, EventType.CYCLE_END)) == 3


async def test_max_cycles_zero_runs_until_a_halt_condition(
    settings: Settings, bus: EventBus
) -> None:
    unbounded = build_settings(settings.workspace, max_cycles=0, max_stalled_cycles=2)
    client = UnusableWorkerClient(unbounded, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, unbounded, bus)
    summary = await run_within(orchestrator)
    assert summary.halted_reason == "stalled"
    assert summary.cycles == 3


async def test_an_empty_plan_halts_as_goal_complete(
    settings: Settings, bus: EventBus
) -> None:
    client = ScriptedLLMClient(settings, plan=plan_of(), test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, settings, bus)
    summary = await run_within(orchestrator)
    assert summary.cycles == 1
    assert summary.halted_reason == "goal_complete"
    assert orchestrator.tasks == {}
    assert any("goal appears complete" in event.message for event in events_of(bus, EventType.LOG))


# ------------------------------------------------------------------- stall path
async def test_a_static_plan_that_completes_nothing_stalls(
    settings: Settings, bus: EventBus
) -> None:
    # Regression: a plan that makes no progress used to spin forever instead of halting.
    stalling = build_settings(settings.workspace, max_cycles=0, max_stalled_cycles=2)
    client = UnusableWorkerClient(stalling, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, stalling, bus)
    summary = await asyncio.wait_for(orchestrator.run(), timeout=RUN_TIMEOUT)
    assert summary.halted_reason == "stalled"
    assert summary.tasks_total == 1
    assert summary.tasks_completed == 0
    assert summary.tasks_failed == 1
    assert events_of(bus, EventType.CYCLE_START)


async def test_a_stalled_run_reports_the_stall_count_on_every_cycle(
    settings: Settings, bus: EventBus
) -> None:
    stalling = build_settings(settings.workspace, max_cycles=0, max_stalled_cycles=3)
    client = UnusableWorkerClient(stalling, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, stalling, bus)
    summary = await run_within(orchestrator)
    ends = events_of(bus, EventType.CYCLE_END)
    assert summary.halted_reason == "stalled"
    assert [event.data["stalled_cycles"] for event in ends] == [0, 1, 2, 3]


# ----------------------------------------------------------------- error paths
async def test_a_raising_client_does_not_propagate_out_of_run(
    settings: Settings, bus: EventBus
) -> None:
    # Regression: the planner call was unguarded, so one API error killed the process.
    failing = build_settings(settings.workspace, max_cycles=0, max_stalled_cycles=2)
    client = ScriptedLLMClient(
        failing, plan=DEFAULT_PLAN, test=DEFAULT_TEST, error=RuntimeError("api is down")
    )
    orchestrator = make_orchestrator(GOAL, client, failing, bus)
    summary = await run_within(orchestrator)
    assert summary.halted_reason == "stalled"
    assert client.error_calls == summary.cycles  # the planner raised once per cycle
    errors = events_of(bus, EventType.ERROR)
    assert errors and all("api is down" in event.message for event in errors)
    assert all(event.data["context"] == "Planner failed" for event in errors)


async def test_a_failing_tester_does_not_propagate_out_of_run(
    settings: Settings, bus: EventBus
) -> None:
    with_failing_suite = build_settings(
        settings.workspace, test_command=FAILING_COMMAND, max_cycles=1
    )
    client = TesterDownClient(with_failing_suite, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, with_failing_suite, bus)
    summary = await run_within(orchestrator)
    assert summary.cycles == 1
    assert orchestrator.test_results == []
    errors = events_of(bus, EventType.ERROR)
    assert any(event.data["context"] == "Tester failed" for event in errors)


async def test_run_start_and_run_end_are_emitted_on_the_error_path(
    settings: Settings, bus: EventBus
) -> None:
    broken = build_settings(settings.workspace, max_cycles=1)
    client = ScriptedLLMClient(broken, error=RuntimeError("nope"))
    orchestrator = make_orchestrator(GOAL, client, broken, bus)
    await run_within(orchestrator)
    assert len(events_of(bus, EventType.RUN_START)) == 1
    ends = events_of(bus, EventType.RUN_END)
    assert len(ends) == 1
    assert ends[0].data["halted_reason"] == orchestrator.halted_reason


async def test_run_start_and_run_end_bracket_a_normal_run(
    settings: Settings, bus: EventBus
) -> None:
    once = build_settings(settings.workspace, max_cycles=1)
    client = ScriptedLLMClient(once, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, once, bus)
    summary = await run_within(orchestrator)
    kinds = [event.type for event in bus.history()]
    assert kinds[0] is EventType.RUN_START
    assert kinds[-1] is EventType.RUN_END
    assert summary.halted_reason == "max_cycles"


# ------------------------------------------------------------------ goal safety
async def test_the_goal_is_never_overwritten_by_test_failures(
    settings: Settings, bus: EventBus
) -> None:
    # Regression: the orchestrator assigned the test summary to self.goal, so the planner
    # forgot the objective from the second cycle onwards.
    failing_suite = build_settings(
        settings.workspace,
        test_command=FAILING_COMMAND,
        max_cycles=3,
        max_stalled_cycles=99,
    )
    client = UnusableWorkerClient(failing_suite, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, failing_suite, bus)
    await run_within(orchestrator)
    assert orchestrator.goal == GOAL
    assert orchestrator.snapshot()["goal"] == GOAL
    # The feedback path still runs, which is what proves the two fields are distinct.
    assert orchestrator.focus != GOAL
    assert "Current failing tests" in orchestrator.focus


async def test_the_planner_keeps_receiving_the_original_goal(
    settings: Settings, bus: EventBus
) -> None:
    failing_suite = build_settings(
        settings.workspace, test_command=FAILING_COMMAND, max_cycles=2, max_stalled_cycles=99
    )
    client = UnusableWorkerClient(failing_suite, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, failing_suite, bus)
    await run_within(orchestrator)
    planner_prompts = [
        call["prompt"]
        for call in client.calls
        if call["method"] == "complete_json" and call["system"] is not None
        and "planning agent" in call["system"]
    ]
    assert len(planner_prompts) == 2
    assert all(GOAL in prompt for prompt in planner_prompts)
    assert all("Current failing tests" in prompt for prompt in planner_prompts[1:])


# ------------------------------------------------------------------- scheduling
async def test_peak_concurrency_never_exceeds_max_parallel_tasks(
    settings: Settings, bus: EventBus
) -> None:
    # Regression: max_parallel_tasks was never enforced, so every task ran at once.
    limited = build_settings(settings.workspace, max_parallel_tasks=2, max_cycles=1)
    tasks = plan_of(*[task_of(str(index)) for index in range(1, 9)])
    client = ScriptedLLMClient(limited, plan=tasks, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, limited, bus)
    summary = await run_within(orchestrator)
    assert summary.tasks_total == 8
    assert summary.tasks_completed == 8
    assert peak_concurrency(bus) == 2
    assert len(task_events(bus, EventType.TASK_START)) == 8
    assert len(task_events(bus, EventType.TASK_END)) == 8
    intervals = worker_intervals(bus)
    assert len(intervals) == 8
    assert all(start <= end for start, end in intervals.values())


async def test_serialised_dependencies_run_in_order(
    settings: Settings, bus: EventBus
) -> None:
    # Regression: TaskModel.dependencies was never read, so dependent tasks ran concurrently.
    serial = build_settings(settings.workspace, max_parallel_tasks=5, max_cycles=1)
    tasks = plan_of(task_of("a"), task_of("b", ["a"]), task_of("c", ["a"]))
    client = ScriptedLLMClient(serial, plan=tasks, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, serial, bus)
    await run_within(orchestrator)
    ends = {event.data["id"]: event for event in task_events(bus, EventType.TASK_END)}
    starts = {event.data["id"]: event for event in task_events(bus, EventType.TASK_START)}
    assert ends["a"].timestamp <= starts["b"].timestamp
    assert ends["a"].timestamp <= starts["c"].timestamp
    order = [event.data["id"] for event in task_events(bus, EventType.TASK_START)]
    assert order[0] == "a"


async def test_an_unrelated_task_may_run_in_the_first_wave(
    settings: Settings, bus: EventBus
) -> None:
    parallel = build_settings(settings.workspace, max_parallel_tasks=5, max_cycles=1)
    tasks = plan_of(task_of("a"), task_of("b", ["a"]), task_of("free"))
    client = ScriptedLLMClient(parallel, plan=tasks, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, parallel, bus)
    await run_within(orchestrator)
    # Wave order is the order TASK_START is emitted in: only "a" and "free" are unblocked.
    started = [event.data["id"] for event in task_events(bus, EventType.TASK_START)]
    assert started == ["a", "free", "b"]


async def test_unknown_dependencies_are_tolerated(
    settings: Settings, bus: EventBus
) -> None:
    with_ghost = build_settings(settings.workspace, max_cycles=1)
    tasks = plan_of(task_of("a", ["ghost-task"]))
    client = ScriptedLLMClient(with_ghost, plan=tasks, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, with_ghost, bus)
    summary = await run_within(orchestrator)
    assert summary.tasks_completed == 1
    assert events_of(bus, EventType.TASK_END)[0].data["status"] == "completed"
    warnings = [event for event in events_of(bus, EventType.LOG) if "Unknown" in event.message]
    assert warnings and warnings[0].data["unknown_dependencies"] == ["ghost-task"]


async def test_a_mutual_dependency_cycle_still_runs(
    settings: Settings, bus: EventBus
) -> None:
    deadlocked = build_settings(settings.workspace, max_cycles=1)
    tasks = plan_of(task_of("a", ["b"]), task_of("b", ["a"]))
    client = ScriptedLLMClient(deadlocked, plan=tasks, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, deadlocked, bus)
    summary = await run_within(orchestrator)
    assert summary.tasks_completed == 2
    logs = events_of(bus, EventType.LOG)
    assert any("Dependency cycle detected" in event.message for event in logs)


def test_plan_waves_breaks_a_self_dependency_without_looping(
    settings: Settings, bus: EventBus
) -> None:
    # A self-dependency can only reach the orchestrator from a hand-built task, because the
    # planner sanitises it away; the wave planner must still terminate on one.
    orchestrator = make_orchestrator(GOAL, ScriptedLLMClient(settings), settings, bus)
    task = TaskModel(id="a", description="d", dependencies=["a"])
    waves = orchestrator._plan_waves([task])
    assert [[item.id for item in wave] for wave in waves] == [["a"]]


# ---------------------------------------------------------------------- summary
async def test_run_summary_counts_completed_and_failed_tasks(
    settings: Settings, bus: EventBus
) -> None:
    once = build_settings(settings.workspace, test_command=PASSING_COMMAND, max_cycles=1)
    tasks = plan_of(task_of("a"), task_of("b"))
    client = ScriptedLLMClient(once, plan=tasks, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, once, bus)
    summary = await run_within(orchestrator)
    assert (summary.cycles, summary.tasks_total) == (1, 2)
    assert (summary.tasks_completed, summary.tasks_failed) == (2, 0)
    assert summary.tests_passed is True
    assert summary.halted_reason == "max_cycles"


async def test_a_failing_suite_overrides_a_false_model_pass(
    settings: Settings, bus: EventBus
) -> None:
    # Regression: the tester used `passed="passed" in response.lower()` on a parse failure,
    # which reported a pass for "0 passed, 3 failed".
    failing = build_settings(
        settings.workspace, test_command=FAILING_COMMAND, max_cycles=1
    )
    client = ScriptedLLMClient(
        failing, plan=DEFAULT_PLAN, test={"passed": True, "summary": "all good", "failures": []}
    )
    orchestrator = make_orchestrator(GOAL, client, failing, bus)
    summary = await run_within(orchestrator)
    assert summary.tests_passed is False
    assert orchestrator.test_results[0].passed is False
    assert any("exited with code 1" in failure for failure in orchestrator.test_results[0].failures)


async def test_test_commands_run_inside_the_workspace(
    settings: Settings, bus: EventBus, workspace: Path
) -> None:
    marker = workspace / "probe.txt"
    marker.write_text("x", encoding="utf-8")
    probe = build_settings(
        settings.workspace,
        test_command=command(
            "import pathlib, sys; sys.exit(0 if pathlib.Path('probe.txt').exists() else 3)"
        ),
        max_cycles=1,
    )
    client = ScriptedLLMClient(probe, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, probe, bus)
    summary = await run_within(orchestrator)
    assert summary.tests_passed is True
    assert orchestrator.test_results[0].commands


async def test_the_tester_is_not_consulted_when_no_runner_is_detected(
    settings: Settings, bus: EventBus
) -> None:
    # tmp_path has no project markers, so no subprocess may be spawned at all.
    client = ScriptedLLMClient(settings, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, settings, bus)
    await run_within(orchestrator)
    assert orchestrator.test_results[0].commands == []
    assert orchestrator.test_results[0].passed is True
    assert not [call for call in client.calls if call["system"] == TESTER_PROMPT]


# ---------------------------------------------------------------------- workers
async def test_a_worker_tool_call_really_writes_into_the_workspace(
    settings: Settings, bus: EventBus, workspace: Path
) -> None:
    once = build_settings(settings.workspace, max_cycles=1)
    rounds = [
        [ToolCall(id="t1", name="write_file", arguments={"path": "hello.txt", "content": "hi"})],
        [],
    ]
    client = ScriptedLLMClient(once, plan=DEFAULT_PLAN, test=DEFAULT_TEST, tool_rounds=rounds)
    orchestrator = make_orchestrator(GOAL, client, once, bus)
    await run_within(orchestrator)
    assert (workspace / "hello.txt").read_text(encoding="utf-8") == "hi"
    tool_events = [event for event in events_of(bus, EventType.TOOL_CALL)]
    assert [event.data["tool"] for event in tool_events][-1] == "write_file"
    assert orchestrator.tasks["1"].status.value == "completed"
    assert "hello.txt" in (orchestrator.tasks["1"].result or "")


async def test_a_failing_worker_marks_the_task_failed_without_raising(
    settings: Settings, bus: EventBus
) -> None:
    once = build_settings(settings.workspace, max_cycles=1)
    client = UnusableWorkerClient(once, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, once, bus)
    summary = await run_within(orchestrator)
    task = orchestrator.tasks["1"]
    assert task.status.value == "failed"
    assert "broken" in (task.result or "")
    assert summary.tasks_failed == 1
    assert [event.data["status"] for event in task_events(bus, EventType.TASK_END)] == ["failed"]


# ------------------------------------------------------------------ task state
async def test_task_state_persists_and_accumulates_across_cycles(
    settings: Settings, bus: EventBus
) -> None:
    twice = build_settings(settings.workspace, max_cycles=2)
    client = ScriptedLLMClient(twice, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    client.queue_plan(plan_of(task_of("1", description="do 1 the second way")))
    orchestrator = make_orchestrator(GOAL, client, twice, bus)
    summary = await run_within(orchestrator)
    task = orchestrator.tasks["1"]
    assert summary.cycles == 2
    # `Orchestrator._run_task` and `WorkerAgent.run` each increment attempts per cycle.
    assert task.attempts == 2 * summary.cycles
    assert task.status.value == "completed"
    # A re-planned id updates the description instead of resetting the task.
    assert task.description == "do 1 the second way"


async def test_a_newly_planned_task_is_added_to_the_running_set(
    settings: Settings, bus: EventBus
) -> None:
    twice = build_settings(settings.workspace, max_cycles=2)
    client = ScriptedLLMClient(twice, plan=plan_of(task_of("1")), test=DEFAULT_TEST)
    client.queue_plan(plan_of(task_of("1"), task_of("2")))
    orchestrator = make_orchestrator(GOAL, client, twice, bus)
    summary = await run_within(orchestrator)
    assert set(orchestrator.tasks) == {"1", "2"}
    assert summary.tasks_total == 2
    assert any(event.data.get("new_tasks") == 1 for event in events_of(bus, EventType.LOG))


# --------------------------------------------------------------------- stopping
async def test_stop_halts_the_run(settings: Settings, bus: EventBus) -> None:
    forever = build_settings(settings.workspace, max_cycles=0, max_stalled_cycles=99)
    client = GatedPlannerClient(forever, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, forever, bus)
    running = asyncio.create_task(orchestrator.run())
    await asyncio.wait_for(client.entered.wait(), timeout=10)
    await orchestrator.stop()
    client.release.set()
    summary = await asyncio.wait_for(running, timeout=RUN_TIMEOUT)
    assert summary.halted_reason == "stopped"
    assert orchestrator.halted_reason == "stopped"
    assert summary.cycles == 1


async def test_a_second_stop_is_harmless(settings: Settings, bus: EventBus) -> None:
    once = build_settings(settings.workspace, max_cycles=1)
    orchestrator = make_orchestrator(GOAL, ScriptedLLMClient(once, plan=DEFAULT_PLAN), once, bus)
    await orchestrator.stop()
    await orchestrator.stop()
    summary = await run_within(orchestrator)
    assert summary.halted_reason == "stopped"
    assert summary.cycles == 0


# --------------------------------------------------------------------- snapshot
async def test_snapshot_is_json_serialisable_and_complete(
    settings: Settings, bus: EventBus
) -> None:
    once = build_settings(settings.workspace, max_cycles=1)
    client = ScriptedLLMClient(once, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, once, bus)
    await run_within(orchestrator)
    snapshot = orchestrator.snapshot()
    assert {
        "goal",
        "cycles",
        "tasks",
        "test_results",
        "running",
        "halted_reason",
    } <= set(snapshot)
    assert snapshot["goal"] == GOAL
    assert snapshot["cycles"] == 1
    assert snapshot["running"] is False
    assert snapshot["halted_reason"] == "max_cycles"
    assert json.loads(json.dumps(snapshot))["tasks"][0]["id"] == "1"


async def test_snapshot_reports_running_while_the_run_is_in_flight(
    settings: Settings, bus: EventBus
) -> None:
    forever = build_settings(settings.workspace, max_cycles=0, max_stalled_cycles=99)
    client = GatedPlannerClient(forever, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, forever, bus)
    running = asyncio.create_task(orchestrator.run())
    await asyncio.wait_for(client.entered.wait(), timeout=10)
    assert orchestrator.snapshot()["running"] is True
    await orchestrator.stop()
    client.release.set()
    await asyncio.wait_for(running, timeout=RUN_TIMEOUT)
    assert orchestrator.snapshot()["running"] is False


async def test_snapshot_exposes_the_client_usage(settings: Settings, bus: EventBus) -> None:
    once = build_settings(settings.workspace, max_cycles=1)
    client = ScriptedLLMClient(once, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, once, bus)
    await run_within(orchestrator)
    usage = orchestrator.snapshot()["usage"]
    assert usage["requests"] == len(client.calls)
    assert usage["input_tokens"] > 0


# ------------------------------------------------------------------- state file
async def test_state_file_is_written_atomically_after_every_cycle(
    settings: Settings, bus: EventBus, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_file = workspace / "state" / "run.json"
    persisted = build_settings(settings.workspace, state_file=state_file, max_cycles=2)
    client = ScriptedLLMClient(persisted, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, persisted, bus)
    replacements: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src: Any, dst: Any) -> None:
        replacements.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(orchestrator_module.os, "replace", spy)
    summary = await run_within(orchestrator)
    assert len(replacements) >= summary.cycles
    assert all(src.endswith(".tmp") for src, _ in replacements)
    assert all(dst == str(state_file) for _, dst in replacements)
    assert state_file.exists()
    assert list(state_file.parent.glob("*.tmp")) == []
    assert json.loads(state_file.read_text(encoding="utf-8"))["cycles"] == summary.cycles


async def test_state_file_round_trips_the_snapshot(
    settings: Settings, bus: EventBus, workspace: Path
) -> None:
    state_file = workspace / "state.json"
    persisted = build_settings(settings.workspace, state_file=state_file, max_cycles=1)
    client = ScriptedLLMClient(persisted, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, persisted, bus)
    await run_within(orchestrator)
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved == orchestrator.snapshot()
    assert saved["goal"] == GOAL


async def test_a_broken_state_path_is_reported_not_raised(
    settings: Settings, bus: EventBus
) -> None:
    unwritable = build_settings(
        settings.workspace, state_file=Path("/proc/cannot/write.json"), max_cycles=1
    )
    client = ScriptedLLMClient(unwritable, plan=DEFAULT_PLAN, test=DEFAULT_TEST)
    orchestrator = make_orchestrator(GOAL, client, unwritable, bus)
    summary = await run_within(orchestrator)
    assert summary.cycles == 1
    assert any(
        event.data.get("context") == "State persistence failed"
        for event in events_of(bus, EventType.ERROR)
    )
