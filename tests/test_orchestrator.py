"""Tests for :mod:`furrow.core.orchestrator`.

Regression suite for the P0 defect that shipped: ``_get_tasks()`` returned
``[]``, so ``_is_done()`` evaluated ``0 >= 0`` and the "infinite" loop halted
after a single cycle.  Every test here therefore drives the loop to a scripted
planner with a finite script, and an unexpected extra ``plan()`` call raises
``AssertionError`` instead of hanging.
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

EventList = list[tuple[str, dict[str, Any]]]


def make_client(*, max_cycles: int = 0, max_parallel_tasks: int = 2) -> LLMClient:
    """A real LLMClient with hermetic settings (no LLM call is ever made)."""
    return LLMClient(
        Settings(
            _env_file=None,
            max_cycles=max_cycles,
            max_parallel_tasks=max_parallel_tasks,
            anthropic_api_key="unused",
            openai_api_key="unused",
            workspace=Path("/tmp"),
        )
    )


def make_plan(*task_ids: str) -> Plan:
    return Plan(
        tasks=[
            TaskModel(id=task_id, description=f"do {task_id}", files=[f"{task_id}.py"])
            for task_id in task_ids
        ],
        rationale="scripted",
    )


def failing(summary: str = "1 failed") -> TestResult:
    return TestResult(passed=False, summary=summary, failures=["tests/test_a.py::test_x"])


def passing() -> TestResult:
    return TestResult(passed=True, summary="2 passed", failures=[])


def sink(events: EventList) -> Callable[[str, dict[str, Any]], None]:
    """An EventSink that proves every payload is JSON-serialisable on arrival."""

    def record(event: str, payload: dict[str, Any]) -> None:
        json.dumps(payload)  # raises TypeError if the orchestrator emitted a non-JSON value
        events.append((event, payload))

    return record


@pytest.fixture(autouse=True)
def _quiet_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the rich output (and its live-refresh machinery) off stdout."""
    monkeypatch.setattr(
        orchestrator_module, "console", Console(file=io.StringIO(), width=120)
    )


class ScriptedPlanner:
    """Stands in for ``PlannerAgent``; the script is finite on purpose."""

    def __init__(self, plans: list[Plan]) -> None:
        self.plans = list(plans)
        self.goals: list[str] = []

    async def plan(self, goal: str) -> Plan:
        if not self.plans:
            raise AssertionError(
                f"planner.plan() was called {len(self.goals) + 1} time(s), but the script "
                f"only has {len(self.goals)} plan(s): the orchestrator is over-looping"
            )
        self.goals.append(goal)
        return self.plans.pop(0)


class ScriptedOrchestrator(Orchestrator):
    """Overrides the ``_run_task`` / ``_run_tests`` seams; no LLM is used."""

    def __init__(
        self,
        *,
        goal: str,
        client: LLMClient,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        plans: list[Plan],
        test_results: list[TestResult],
        run_task: Callable[[TaskModel], Any] | None = None,
    ) -> None:
        super().__init__(goal=goal, client=client, on_event=on_event)
        self.scripted_planner = ScriptedPlanner(plans)
        self.planner = self.scripted_planner
        self._test_results = list(test_results)
        self._run_task_impl = run_task
        self.task_calls: list[TaskModel] = []

    async def _run_task(self, task: TaskModel) -> str:
        self.task_calls.append(task)
        if self._run_task_impl is not None:
            return await self._run_task_impl(task)
        return f"done: {task.id}"

    async def _run_tests(self, plan: Plan) -> TestResult:
        if not self._test_results:
            raise AssertionError("_run_tests() was called more times than the script allows")
        return self._test_results.pop(0)


class ConcurrencyProbe:
    """A ``_run_task`` implementation that records peak overlap.

    Each task parks on ``gate`` until the driver opens it, so overlap is real
    (no sleeping): the driver waits until ``limit`` tasks have actually started
    before releasing them.  ``peak`` is the high-water mark of concurrently
    running tasks.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.gate = asyncio.Event()
        self.wave_ready = asyncio.Event()
        self.active = 0
        self.peak = 0
        self.started: list[str] = []

    async def __call__(self, task: TaskModel) -> str:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.append(task.id)
        if len(self.started) >= self.limit:
            self.wave_ready.set()
        try:
            await self.gate.wait()
        finally:
            self.active -= 1
        return f"done: {task.id}"


def halts(events: EventList) -> list[dict[str, Any]]:
    return [payload for event, payload in events if event == "halt"]


def payloads(events: EventList, event: str) -> list[dict[str, Any]]:
    return [payload for name, payload in events if name == event]


async def test_failing_then_passing_tests_run_exactly_two_cycles() -> None:
    """P0 regression: cycle 1 fails, cycle 2 passes -> exactly 2 cycles."""
    events: EventList = []
    orchestrator = ScriptedOrchestrator(
        goal="ship it",
        client=make_client(max_cycles=0),
        on_event=sink(events),
        plans=[make_plan("t1", "t2"), make_plan("t1", "t2")],
        test_results=[failing(), passing()],
    )

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert [payload["passed"] for payload in payloads(events, "test_result")] == [False, True]
    assert [h["reason"] for h in halts(events)] == ["goal_complete"]
    assert halts(events)[0]["cycles"] == 2
    # Cycle 2 planned against the accumulated failures, and nothing ran more than once.
    assert len(orchestrator.scripted_planner.goals) == 2
    assert "tests/test_a.py::test_x" in orchestrator.scripted_planner.goals[1]
    assert orchestrator.failures == ["tests/test_a.py::test_x"]
    assert orchestrator.goal == "ship it"
    assert orchestrator.plan is not None
    assert [task.status for task in orchestrator.plan.tasks] == ["completed", "completed"]


async def test_perpetual_failure_stops_at_max_cycles() -> None:
    max_cycles = 3
    events: EventList = []
    orchestrator = ScriptedOrchestrator(
        goal="never converges",
        client=make_client(max_cycles=max_cycles),
        on_event=sink(events),
        plans=[make_plan("t1") for _ in range(max_cycles)],
        test_results=[failing(f"failure {index}") for index in range(max_cycles)],
    )

    await orchestrator.run()

    assert orchestrator.cycles == max_cycles
    assert [h["reason"] for h in halts(events)] == ["max_cycles"]
    assert halts(events)[0]["cycles"] == max_cycles
    assert halts(events)[0]["max_cycles"] == max_cycles
    assert len(payloads(events, "cycle_start")) == max_cycles
    assert len(payloads(events, "retry_planned")) == max_cycles
    assert orchestrator.last_test_result is not None
    assert orchestrator.last_test_result.passed is False


async def test_zero_task_plan_is_not_done_and_does_not_crash() -> None:
    events: EventList = []
    orchestrator = ScriptedOrchestrator(
        goal="nothing to do",
        client=make_client(max_cycles=2),
        on_event=sink(events),
        plans=[make_plan(), make_plan("t1")],
        test_results=[passing()],
    )

    await orchestrator.run()

    assert orchestrator.cycles == 2
    first_cycle_end = payloads(events, "cycle_end")[0]
    assert first_cycle_end["task_count"] == 0
    assert first_cycle_end["done"] is False
    assert [h["reason"] for h in halts(events)] == ["goal_complete"]


async def test_completed_tasks_with_failing_tests_force_another_cycle() -> None:
    events: EventList = []
    first_plan = make_plan("t1", "t2")
    orchestrator = ScriptedOrchestrator(
        goal="tests still red",
        client=make_client(max_cycles=0),
        on_event=sink(events),
        plans=[first_plan, make_plan("t1", "t2")],
        test_results=[TestResult(passed=False, summary="boom", failures=["boom"]), passing()],
    )

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert [task.status for task in first_plan.tasks] == ["completed", "completed"]
    assert payloads(events, "cycle_end")[0]["passed"] is False
    assert [payload["failure"] for payload in payloads(events, "test_failure")] == ["boom"]
    assert orchestrator.last_test_result is not None
    assert orchestrator.last_test_result.passed is True


async def test_task_fan_out_never_exceeds_max_parallel_tasks() -> None:
    max_parallel_tasks = 2
    probe = ConcurrencyProbe(max_parallel_tasks)
    events: EventList = []
    orchestrator = ScriptedOrchestrator(
        goal="fan out",
        client=make_client(max_cycles=1, max_parallel_tasks=max_parallel_tasks),
        on_event=sink(events),
        plans=[make_plan("t1", "t2", "t3", "t4", "t5")],
        test_results=[passing()],
        run_task=probe,
    )

    runner = asyncio.create_task(orchestrator.run())
    first_wave = asyncio.create_task(probe.wave_ready.wait())
    # FIRST_COMPLETED keeps this deadlock-free if the fan-out never fills a wave.
    done, _pending = await asyncio.wait({runner, first_wave}, return_when=asyncio.FIRST_COMPLETED)
    if first_wave not in done:
        probe.gate.set()  # the run finished early; let the assertions speak
    else:
        assert probe.peak == max_parallel_tasks
    probe.gate.set()
    await runner

    assert sorted(probe.started) == ["t1", "t2", "t3", "t4", "t5"]
    assert probe.peak == max_parallel_tasks
    assert probe.active == 0
    assert payloads(events, "parallel_limit")[0] == {
        "event": "parallel_limit",
        "message": "Throttling 5 tasks to 2 concurrent workers.",
        "total": 5,
        "limit": 2,
    }


async def test_emitted_events_are_json_serializable() -> None:
    events: EventList = []
    orchestrator = ScriptedOrchestrator(
        goal="stream me",
        client=make_client(max_cycles=1),
        on_event=sink(events),
        plans=[make_plan("t1", "t2")],
        test_results=[failing()],
    )

    await orchestrator.run()

    names = [event for event, _ in events]
    assert {
        "start",
        "plan",
        "task_start",
        "task_complete",
        "test_result",
        "test_failure",
        "retry_planned",
        "cycle_end",
        "halt",
    } <= set(names)
    assert names[0] == "start"
    assert events[0][1]["goal"] == "stream me"
    # Re-serialising every payload outside the sink proves the values are plain JSON types.
    for event, payload in events:
        assert set(payload) >= {"event", "message"}
        assert payload["event"] == event
        assert json.loads(json.dumps(payload))["message"] == payload["message"]
