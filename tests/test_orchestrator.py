"""Offline tests for the Orchestrator cycle loop.

No network, no real LLMClient and no conftest: the planner, the tester and the
worker seam are all stubbed so the loop mechanics can be asserted
deterministically.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator

WS = Path(".")


class FakeLLM:
    """Offline stand-in for LLMClient; any real call is a test bug."""

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        raise AssertionError("FakeLLM.complete must never be called")

    async def read_file(self, path: str) -> str:
        raise AssertionError("FakeLLM.read_file must never be called")

    async def write_file(self, path: str, content: str) -> None:
        raise AssertionError("FakeLLM.write_file must never be called")

    def list_files(self, directory: str) -> list[str]:
        return []


class StubPlanner:
    def __init__(self, plans: list[Plan]) -> None:
        self._plans = list(plans)
        self.calls: list[str] = []

    async def plan(self, goal: str) -> Plan:
        self.calls.append(goal)
        if len(self._plans) > 1:
            return self._plans.pop(0)
        return self._plans[0]


class StubTester:
    def __init__(self, results: list[TestResult]) -> None:
        self._results = list(results)
        self.calls: list[str] = []

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.calls.append(goal)
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0]


def make_plan(*descriptions: str) -> Plan:
    return Plan(
        tasks=[TaskModel(id=f"t{i}", description=desc) for i, desc in enumerate(descriptions)],
        rationale="stub plan",
    )


PASSING = TestResult(passed=True, summary="all green")
FAILING = TestResult(passed=False, summary="1 failed", failures=["test_a::test_b asserts 2"])


def build(
    goal: str = "ship the feature",
    plans: list[Plan] | None = None,
    results: list[TestResult] | None = None,
    settings: Settings | None = None,
    on_event: Callable[..., Any] | None = None,
) -> Orchestrator:
    orch = Orchestrator(
        goal=goal,
        client=FakeLLM(),
        settings=settings or Settings(max_cycles=10, workspace=WS),
        on_event=on_event,
    )
    orch.planner = StubPlanner(plans or [make_plan("do a thing")])
    orch.tester = StubTester(results or [PASSING])
    return orch


def stub_workers(orch: Orchestrator, handler: Callable[[TaskModel], Any]) -> None:
    """Replace the worker seam with an async ``handler(task)``."""

    async def _run(task: TaskModel) -> Any:
        return await handler(task)

    orch._run_worker = _run  # type: ignore[method-assign]


def record_cycles(orch: Orchestrator) -> list[dict[str, Any]]:
    """Snapshot goal/focus/statuses after every cycle."""
    snapshots: list[dict[str, Any]] = []
    original = orch._cycle

    async def _cycle() -> None:
        await original()
        snapshots.append(
            {
                "cycle": orch.cycles,
                "goal": orch.goal,
                "focus": orch.focus,
                "tests_passed": orch._tests_passed,
                "statuses": [t.status for t in orch.tasks],
            }
        )

    orch._cycle = _cycle  # type: ignore[method-assign]
    return snapshots


async def ok_worker(task: TaskModel) -> Any:
    await asyncio.sleep(0)
    return f"done:{task.id}"


async def test_pending_work_forces_more_cycles_then_completes() -> None:
    events: list[dict[str, Any]] = []
    orch = build(
        plans=[make_plan("first", "second"), make_plan("first", "second")],
        results=[PASSING, PASSING],
        on_event=events.append,
    )
    snapshots = record_cycles(orch)

    async def worker(task: TaskModel) -> Any:
        await asyncio.sleep(0)
        # Cycle 1 defers one task; the orchestrator must keep looping for it.
        if orch.cycles == 1 and task.id == "t1":
            return None
        return f"done:{task.id}"

    stub_workers(orch, worker)
    await orch.run()

    assert orch.cycles > 1, "loop halted while work was still pending"
    assert len(snapshots) == orch.cycles
    assert snapshots[0]["statuses"] == ["completed", "pending"]
    assert snapshots[-1]["statuses"] == ["completed", "completed"]
    assert [e["type"] for e in events if e["type"] == "done"] == ["done"]
    assert events[-1] == {"type": "done", "reason": "complete"}
    assert next(e for e in events if e["type"] == "cycle_end" and e["cycle"] == 1)["done"] is False


async def test_failing_tests_never_mutate_goal_but_set_focus() -> None:
    events: list[dict[str, Any]] = []
    orch = build(
        goal="keep me intact",
        plans=[make_plan("only"), make_plan("only")],
        results=[FAILING, PASSING],
        on_event=events.append,
    )
    snapshots = record_cycles(orch)
    stub_workers(orch, ok_worker)
    await orch.run()

    assert orch.goal == "keep me intact"
    assert [s["goal"] for s in snapshots] == ["keep me intact"] * 2
    # The failure text only lives in focus, for exactly one cycle.
    assert snapshots[0]["focus"].startswith("Fix failing tests:")
    assert "test_a::test_b asserts 2" in snapshots[0]["focus"]
    assert snapshots[1]["focus"] == "keep me intact"

    # No event ever leaks the rewritten goal or the focus text as the goal.
    assert {e["goal"] for e in events if "goal" in e} == {"keep me intact"}


async def test_failure_text_flows_into_next_plan_call() -> None:
    orch = build(
        plans=[make_plan("only"), make_plan("only"), make_plan("only")],
        results=[FAILING, FAILING, PASSING],
    )
    stub_workers(orch, ok_worker)
    await orch.run()

    planner: StubPlanner = orch.planner  # type: ignore[assignment]
    tester: StubTester = orch.tester  # type: ignore[assignment]
    assert planner.calls[0] == "ship the feature"
    assert planner.calls[1].startswith("Fix failing tests:")
    assert planner.calls[2].startswith("Fix failing tests:")
    # The tester always sees the real goal.
    assert tester.calls == ["ship the feature"] * 3
    assert orch.goal == "ship the feature"


async def test_max_cycles_bounds_the_loop() -> None:
    events: list[dict[str, Any]] = []
    orch = build(
        plans=[make_plan("never finishes")],
        settings=Settings(max_cycles=2, workspace=WS),
        on_event=events.append,
    )

    async def worker(task: TaskModel) -> Any:
        return None

    stub_workers(orch, worker)
    await orch.run()

    assert orch.cycles == 2
    assert [e["cycle"] for e in events if e["type"] == "cycle_start"] == [1, 2]
    assert events[-1] == {"type": "done", "reason": "max_cycles"}


async def test_max_parallel_tasks_caps_concurrency() -> None:
    active = 0
    peak = 0

    async def worker(task: TaskModel) -> Any:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return f"done:{task.id}"

    orch = build(
        plans=[make_plan("a", "b", "c")],
        settings=Settings(max_cycles=1, max_parallel_tasks=1, workspace=WS),
    )
    stub_workers(orch, worker)
    await orch.run()

    assert peak == 1, f"observed concurrency {peak}, expected a cap of 1"
    assert all(t.status == "completed" for t in orch.tasks)


async def test_empty_plan_halts_with_no_tasks() -> None:
    events: list[dict[str, Any]] = []
    orch = build(plans=[Plan(tasks=[], rationale="nothing to do")], on_event=events.append)

    async def worker(task: TaskModel) -> Any:
        raise AssertionError("no worker may run for an empty plan")

    stub_workers(orch, worker)
    await orch.run()

    assert orch.cycles == 1
    assert orch._is_done() is False
    assert events[-1] == {"type": "done", "reason": "no_tasks"}


async def test_all_events_are_json_serialisable_and_callback_errors_are_swallowed() -> None:
    events: list[dict[str, Any]] = []
    calls = 0

    async def on_event(event: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        events.append(event)
        if calls % 2 == 0:
            raise RuntimeError("UI exploded")

    orch = build(
        plans=[make_plan("a")],
        settings=Settings(max_cycles=1, workspace=WS),
        on_event=on_event,
    )
    stub_workers(orch, ok_worker)
    await orch.run()

    assert events, "the callback raised before any event was recorded"
    # Every callback exception was swallowed; the run still completed.
    assert calls == 6
    assert [e["type"] for e in events] == [
        "cycle_start",
        "plan",
        "task",
        "test",
        "cycle_end",
        "done",
    ]
    assert events[-1] == {"type": "done", "reason": "complete"}
    for event in events:
        json.dumps(event)
