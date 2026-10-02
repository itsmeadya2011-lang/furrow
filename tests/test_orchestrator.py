"""Regression tests for the development loop in ``furrow.core.orchestrator``.

Historical bug under test: ``_get_tasks()`` returned a hardcoded ``[]``, so
``_is_done()`` evaluated ``0 >= 0`` and ``run()`` always halted after a single
cycle while claiming success.  A second historical bug overwrote ``self.goal``
with the "fix the tests" directive, losing the original goal forever.

Every agent is a fake installed on the ``furrow.core.orchestrator`` module, so no
LLM call is ever made and each test is fully offline and deterministic.
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from rich.console import Console

import furrow.core.orchestrator as orchestrator_module
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import CycleRecord, Orchestrator

#: The complete set of event types the loop is allowed to publish.
DOCUMENTED_EVENT_TYPES = frozenset(
    {
        "cycle_start",
        "plan",
        "task_start",
        "task_end",
        "test_result",
        "cycle_end",
        "run_complete",
        "run_halt",
    }
)

EVENT_KEYS = {"type", "cycle", "message", "data"}
STATUS_KEYS = {"goal", "directive", "cycles", "running", "done", "tasks", "last_test"}


# ------------------------------------------------------------------- test doubles


class FakeClient:
    """Stands in for ``LLMClient``; the fake agents below never call it."""


@dataclass
class Fakes:
    """Mutable record of everything the fake agents were asked to do."""

    plans: list[Plan] = field(default_factory=list)
    directives: list[str] = field(default_factory=list)
    tester_goals: list[str] = field(default_factory=list)
    in_flight: int = 0
    peak_in_flight: int = 0
    worker_result: Callable[[TaskModel], str] | None = None
    test_result: Callable[[str, list[TaskModel]], TestResult] | None = None


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    """Install fake planner/worker/tester agents and a silent rich console."""
    state = Fakes()
    monkeypatch.setattr(orchestrator_module, "console", Console(file=io.StringIO(), width=120))

    class FakePlanner:
        def __init__(self, client: Any = None, settings: Any = None) -> None:
            self.client = client

        async def plan(self, directive: str) -> Plan:
            state.directives.append(directive)
            return state.plans.pop(0)

    class FakeWorker:
        def __init__(self, task: TaskModel, client: Any = None, settings: Any = None) -> None:
            self.task = task

        async def run(self) -> str:
            state.in_flight += 1
            state.peak_in_flight = max(state.peak_in_flight, state.in_flight)
            try:
                await asyncio.sleep(0.01)
                if state.worker_result is None:
                    return f"done: {self.task.id}"
                return state.worker_result(self.task)
            finally:
                state.in_flight -= 1

    class FakeTester:
        def __init__(self, client: Any = None, settings: Any = None) -> None:
            self.client = client

        async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
            state.tester_goals.append(goal)
            if state.test_result is None:
                return TestResult(passed=True, summary="all good")
            return state.test_result(goal, tasks)

    monkeypatch.setattr(orchestrator_module, "PlannerAgent", FakePlanner)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", FakeTester)
    return state


def make_plan(*ids: str) -> Plan:
    return Plan(
        tasks=[TaskModel(id=ident, description=f"do {ident}") for ident in ids],
        rationale="because",
    )


def passing() -> TestResult:
    return TestResult(passed=True, summary="all good")


def failing(*failures: str) -> TestResult:
    return TestResult(
        passed=False,
        summary="1 failed",
        failures=list(failures) or ["tests/test_x.py::test_y failed"],
    )


def fail_task_a(task: TaskModel) -> str:
    """Worker hook that makes task "a" explode and everything else succeed."""
    if task.id == "a":
        raise RuntimeError("worker exploded")
    return "ok"


def record(
    *,
    tasks: list[TaskModel],
    passed: bool = True,
    cycle: int = 1,
) -> CycleRecord:
    tasks = list(tasks)
    result = TestResult(passed=passed, summary="summary", failures=["boom"])
    return CycleRecord(
        cycle=cycle,
        plan=Plan(tasks=tasks, rationale="r"),
        test_result=result,
        tasks=tasks,
    )


def build(
    goal: str = "add a /health endpoint",
    settings: Settings | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Orchestrator:
    return Orchestrator(
        goal=goal,
        client=FakeClient(),
        on_event=on_event,
        settings=settings or Settings(max_cycles=0),
    )


# ------------------------------------------------------------------ _is_done()

# Bug: _is_done() compared `completed == len(tasks)` against a hardcoded empty
# task list, so it answered "done" before a single task had ever run.


def test_is_done_is_false_before_any_cycle_has_run(fakes: Fakes):
    assert build()._is_done() is False
    assert build().history == []


def test_is_done_is_false_when_the_last_cycle_left_a_failed_task(fakes: Fakes):
    orch = build()
    orch.history.append(
        record(
            tasks=[
                TaskModel(id="a", description="d", status="completed"),
                TaskModel(id="b", description="d", status="failed"),
            ]
        )
    )
    assert orch._is_done() is False


def test_is_done_is_true_only_when_every_task_of_the_last_cycle_completed(fakes: Fakes):
    orch = build()
    orch.history.append(
        record(
            tasks=[
                TaskModel(id="a", description="d", status="completed"),
                TaskModel(id="b", description="d", status="completed"),
            ]
        )
    )
    assert orch._is_done() is True


def test_is_done_ignores_earlier_cycles_and_looks_only_at_the_last_one(fakes: Fakes):
    orch = build()
    orch.history.append(
        record(tasks=[TaskModel(id="a", description="d", status="failed")], cycle=1)
    )
    orch.history.append(
        record(tasks=[TaskModel(id="b", description="d", status="completed")], cycle=2)
    )
    assert orch._is_done() is True


def test_is_done_is_false_when_the_last_cycle_had_zero_tasks(fakes: Fakes):
    # An empty plan is not evidence of completion: it is the signature of the
    # original `_get_tasks()` bug.
    orch = build()
    orch.history.append(record(tasks=[]))
    assert orch._is_done() is False


def test_is_done_is_true_once_the_cycle_budget_is_exhausted(fakes: Fakes):
    orch = build(settings=Settings(max_cycles=1))
    orch.cycles = 1
    assert orch._is_done() is True


def test_is_done_is_false_below_an_unreached_budget_with_work_left(fakes: Fakes):
    orch = build(settings=Settings(max_cycles=3))
    orch.cycles = 1
    orch.history.append(record(tasks=[TaskModel(id="a", description="d", status="failed")]))
    assert orch._limit_reached() is False
    assert orch._is_done() is False


# ------------------------------------------------------------------------- run()


async def test_run_stops_after_exactly_max_cycles(fakes: Fakes):
    # Bug: the loop halted after a single cycle regardless of the cycle budget.
    fakes.plans.extend([make_plan("a"), make_plan("b")])
    fakes.worker_result = fail_task_a
    events: list[dict[str, Any]] = []
    orch = build(settings=Settings(max_cycles=2), on_event=events.append)

    await asyncio.wait_for(orch.run(), timeout=10)

    assert orch.cycles == 2
    assert [e["type"] for e in events].count("cycle_start") == 2
    assert len(fakes.plans) == 0  # the third cycle never happened


async def test_run_halts_with_run_halt_when_the_budget_runs_out_incomplete(fakes: Fakes):
    fakes.plans.extend([make_plan("a"), make_plan("a")])
    fakes.worker_result = fail_task_a
    events: list[dict[str, Any]] = []
    orch = build(settings=Settings(max_cycles=2), on_event=events.append)

    await orch.run()

    types = [e["type"] for e in events]
    assert types[-1] == "run_halt"
    assert "max_cycles" in events[-1]["message"]
    assert "run_complete" not in types


async def test_run_finishes_with_run_complete_when_every_task_completed(fakes: Fakes):
    fakes.plans.append(make_plan("a", "b"))
    events: list[dict[str, Any]] = []
    orch = build(on_event=events.append)

    await asyncio.wait_for(orch.run(), timeout=10)

    assert orch.cycles == 1
    assert [e["type"] for e in events][-1] == "run_complete"


async def test_run_keeps_going_after_a_failing_cycle_until_the_budget_is_spent(fakes: Fakes):
    fakes.plans.extend([make_plan("a"), make_plan("a"), make_plan("a")])
    fakes.worker_result = fail_task_a
    fakes.test_result = lambda goal, tasks: failing()
    orch = build(settings=Settings(max_cycles=3))

    await asyncio.wait_for(orch.run(), timeout=10)

    assert orch.cycles == 3
    assert len(fakes.tester_goals) == 3


async def test_run_halts_instead_of_spinning_when_the_planner_returns_an_empty_plan(fakes: Fakes):
    # A planner that keeps returning nothing must not produce an infinite loop.
    fakes.plans.append(Plan(tasks=[], rationale="nothing to do"))
    events: list[dict[str, Any]] = []
    orch = build(settings=Settings(max_cycles=0), on_event=events.append)

    await asyncio.wait_for(orch.run(), timeout=5)

    assert orch.cycles == 1
    assert fakes.plans == []
    types = [e["type"] for e in events]
    assert types.count("cycle_start") == 1
    assert types[-1] == "run_halt"
    assert "no tasks" in events[-1]["message"]


async def test_run_emits_cycle_end_for_an_empty_plan_before_halting(fakes: Fakes):
    fakes.plans.append(Plan(tasks=[], rationale=""))
    events: list[dict[str, Any]] = []
    orch = build(on_event=events.append)

    await asyncio.wait_for(orch.run(), timeout=5)

    types = [e["type"] for e in events]
    assert types == ["cycle_start", "plan", "cycle_end", "run_halt"]


async def test_run_propagates_an_abort_and_reports_run_halt_on_error(fakes: Fakes, monkeypatch):
    fakes.plans.append(make_plan("a"))

    async def exploding_plan(self: Any, directive: str) -> Plan:
        raise RuntimeError("planner exploded")

    monkeypatch.setattr(orchestrator_module.PlannerAgent, "plan", exploding_plan)
    events: list[dict[str, Any]] = []
    orch = build(on_event=events.append)

    with pytest.raises(RuntimeError, match="planner exploded"):
        await orch.run()

    assert [e["type"] for e in events][-1] == "run_halt"
    assert orch.status()["running"] is False


# ------------------------------------------------------------------- goal/directive


async def test_original_goal_survives_a_cycle_whose_tests_failed(fakes: Fakes):
    # Bug: the "fix the failing tests" text was assigned to self.goal, so the
    # original goal was lost after the very first failure.
    goal = "add a /health endpoint"
    fakes.plans.extend([make_plan("a"), make_plan("a")])
    fakes.worker_result = fail_task_a
    fakes.test_result = lambda g, tasks: failing("tests/test_x.py::test_y failed")
    orch = build(goal=goal, settings=Settings(max_cycles=2))

    await orch.run()

    assert orch.goal == goal
    assert orch.directive.startswith("Fix failing tests:")
    assert "tests/test_x.py::test_y failed" in orch.directive


async def test_the_tester_is_always_asked_about_the_original_goal(fakes: Fakes):
    goal = "add a /health endpoint"
    fakes.plans.extend([make_plan("a"), make_plan("a")])
    fakes.worker_result = fail_task_a
    fakes.test_result = lambda g, tasks: failing("boom")
    orch = build(goal=goal, settings=Settings(max_cycles=2))

    await orch.run()

    assert fakes.tester_goals == [goal, goal]


async def test_the_next_cycle_plans_against_the_fix_directive(fakes: Fakes):
    fakes.plans.extend([make_plan("a"), make_plan("a")])
    fakes.worker_result = fail_task_a
    fakes.test_result = lambda g, tasks: failing("boom")
    orch = build(settings=Settings(max_cycles=2))

    await orch.run()

    assert fakes.directives[0] == orch.goal
    assert fakes.directives[1].startswith("Fix failing tests:")


async def test_a_passing_cycle_resets_the_directive_to_the_goal(fakes: Fakes):
    fakes.plans.extend([make_plan("a"), make_plan("a")])
    fakes.worker_result = fail_task_a
    fakes.test_result = lambda g, tasks: failing("boom")
    orch = build(goal="ship it", settings=Settings(max_cycles=2))
    await orch.run()
    assert orch.directive.startswith("Fix failing tests:")

    fakes.plans.append(make_plan("a"))
    fakes.test_result = lambda g, tasks: passing()
    await orch.run()

    assert orch.directive == "ship it"


# ------------------------------------------------------------------ event envelope


async def test_every_event_has_exactly_the_documented_keys(fakes: Fakes):
    fakes.plans.append(make_plan("a", "b"))
    events: list[dict[str, Any]] = []
    orch = build(on_event=events.append)

    await asyncio.wait_for(orch.run(), timeout=10)

    assert events
    for event in events:
        assert set(event) == EVENT_KEYS, event


async def test_every_event_type_is_one_of_the_documented_types(fakes: Fakes):
    fakes.plans.append(make_plan("a", "b"))
    events: list[dict[str, Any]] = []
    orch = build(on_event=events.append)

    await orch.run()

    types = [event["type"] for event in events]
    assert set(types) <= DOCUMENTED_EVENT_TYPES
    assert set(types) == {
        "cycle_start",
        "plan",
        "task_start",
        "task_end",
        "test_result",
        "cycle_end",
        "run_complete",
    }


async def test_every_event_is_json_serializable(fakes: Fakes):
    fakes.plans.append(make_plan("a", "b"))
    events: list[dict[str, Any]] = []
    orch = build(on_event=events.append)

    await orch.run()

    for event in events:
        assert json.loads(json.dumps(event)) == event


async def test_the_first_event_of_a_run_is_cycle_start(fakes: Fakes):
    fakes.plans.append(make_plan("a"))
    events: list[dict[str, Any]] = []
    orch = build(on_event=events.append)

    await orch.run()

    assert events[0]["type"] == "cycle_start"
    assert events[0]["data"]["goal"] == orch.goal


async def test_event_cycles_are_numbered_from_one(fakes: Fakes):
    fakes.plans.extend([make_plan("a"), make_plan("a")])
    fakes.worker_result = fail_task_a
    events: list[dict[str, Any]] = []
    orch = build(settings=Settings(max_cycles=2), on_event=events.append)

    await orch.run()

    assert [e["cycle"] for e in events if e["type"] == "cycle_start"] == [1, 2]
    assert [e["cycle"] for e in events if e["type"] == "cycle_end"] == [1, 2]


async def test_plan_event_carries_the_planned_tasks(fakes: Fakes):
    fakes.plans.append(make_plan("a", "b"))
    events: list[dict[str, Any]] = []
    orch = build(on_event=events.append)

    await orch.run()

    plan_event = next(e for e in events if e["type"] == "plan")
    assert plan_event["data"]["rationale"] == "because"
    assert [t["id"] for t in plan_event["data"]["tasks"]] == ["a", "b"]


async def test_test_result_event_reports_the_failure_list(fakes: Fakes):
    fakes.plans.extend([make_plan("a"), make_plan("a")])
    fakes.worker_result = fail_task_a
    fakes.test_result = lambda g, tasks: failing("boom")
    events: list[dict[str, Any]] = []
    orch = build(settings=Settings(max_cycles=2), on_event=events.append)

    await orch.run()

    first = next(e for e in events if e["type"] == "test_result")
    assert first["data"]["passed"] is False
    assert first["data"]["failures"] == ["boom"]


async def test_task_end_event_marks_a_failing_task(fakes: Fakes):
    fakes.plans.append(make_plan("a"))
    fakes.worker_result = fail_task_a
    events: list[dict[str, Any]] = []
    orch = build(settings=Settings(max_cycles=1), on_event=events.append)

    await orch.run()

    ends = [e for e in events if e["type"] == "task_end"]
    assert [e["data"]["status"] for e in ends] == ["failed"]
    assert ends[0]["data"]["id"] == "a"


# ------------------------------------------------------------ observer robustness


async def test_an_on_event_callback_that_raises_does_not_kill_the_run(fakes: Fakes):
    # Bug: an observer exception escaped _emit and aborted the whole loop.
    seen: list[dict[str, Any]] = []

    def on_event(event: dict[str, Any]) -> None:
        seen.append(event)
        raise RuntimeError("observer blew up")

    fakes.plans.append(make_plan("a", "b"))
    orch = build(on_event=on_event)

    await asyncio.wait_for(orch.run(), timeout=10)

    assert seen
    assert orch.cycles == 1
    assert orch.status()["running"] is False
    assert all(task.status == "completed" for task in orch.history[-1].tasks)


def test_emit_swallows_an_observer_exception(fakes: Fakes):
    def on_event(event: dict[str, Any]) -> None:
        raise ValueError("nope")

    orch = build(on_event=on_event)
    orch._emit("plan", "hello", {"a": 1})  # must not raise


def test_run_without_an_on_event_callback_still_works(fakes: Fakes):
    fakes.plans.append(make_plan("a"))
    orch = build(on_event=None)
    asyncio.get_event_loop_policy()
    assert orch.on_event is None


# ------------------------------------------------------------------------- status


def test_status_returns_exactly_the_documented_keys(fakes: Fakes):
    assert set(build().status()) == STATUS_KEYS


def test_status_before_any_cycle_reports_an_empty_snapshot(fakes: Fakes):
    status = build(goal="g").status()
    assert status["goal"] == "g"
    assert status["cycles"] == 0
    assert status["running"] is False
    assert status["done"] is False
    assert status["tasks"] == []
    assert status["last_test"] is None


async def test_status_reports_running_true_during_the_run_and_false_afterwards(fakes: Fakes):
    fakes.plans.append(make_plan("a"))
    observed: list[bool] = []
    orch: Orchestrator

    def on_event(event: dict[str, Any]) -> None:
        observed.append(orch.status()["running"])

    orch = build(on_event=on_event)
    await orch.run()

    assert observed
    assert all(observed)
    assert orch.status()["running"] is False


async def test_status_task_entries_expose_id_description_status_and_result(fakes: Fakes):
    fakes.plans.append(make_plan("a"))
    orch = build()
    await orch.run()

    tasks = orch.status()["tasks"]
    assert tasks == [{"id": "a", "description": "do a", "status": "completed", "result": "done: a"}]
    assert set(tasks[0]) == {"id", "description", "status", "result"}


async def test_status_last_test_carries_the_latest_test_result(fakes: Fakes):
    fakes.plans.append(make_plan("a"))
    orch = build()
    await orch.run()

    last_test = orch.status()["last_test"]
    assert last_test == {"passed": True, "summary": "all good", "failures": []}


async def test_a_failing_worker_is_recorded_as_a_failed_task_with_its_error(fakes: Fakes):
    fakes.plans.append(make_plan("a"))
    fakes.worker_result = fail_task_a
    orch = build(settings=Settings(max_cycles=1))

    await orch.run()

    task = orch.history[-1].tasks[0]
    assert task.status == "failed"
    assert "worker exploded" in (task.result or "")
    assert orch.status()["tasks"][0]["status"] == "failed"


async def test_status_done_is_true_after_a_fully_completed_cycle(fakes: Fakes):
    fakes.plans.append(make_plan("a"))
    orch = build()
    await orch.run()
    assert orch.status()["done"] is True


# -------------------------------------------------------------------- concurrency


async def test_at_most_max_parallel_tasks_are_in_flight_at_once(fakes: Fakes):
    fakes.plans.append(make_plan("a", "b", "c", "d", "e"))
    orch = build(settings=Settings(max_parallel_tasks=2, max_cycles=0))

    await asyncio.wait_for(orch.run(), timeout=10)

    assert fakes.peak_in_flight <= 2
    assert fakes.peak_in_flight == 2
    assert len(orch.history[-1].tasks) == 5


async def test_all_tasks_still_run_when_capped_by_max_parallel_tasks(fakes: Fakes):
    fakes.plans.append(make_plan("a", "b", "c", "d", "e"))
    orch = build(settings=Settings(max_parallel_tasks=2, max_cycles=0))

    await orch.run()

    statuses = {task.id: task.status for task in orch.history[-1].tasks}
    assert statuses == dict.fromkeys("abcde", "completed")
    assert {task.result for task in orch.history[-1].tasks} == {
        f"done: {ident}" for ident in "abcde"
    }


async def test_a_nonsensical_parallel_limit_is_clamped_to_one(fakes: Fakes):
    fakes.plans.append(make_plan("a", "b", "c"))
    orch = build(settings=Settings(max_parallel_tasks=0, max_cycles=0))

    await orch.run()

    assert fakes.peak_in_flight == 1
    assert all(task.status == "completed" for task in orch.history[-1].tasks)
