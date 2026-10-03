"""Regression suite for furrow.core.orchestrator.Orchestrator.

Every collaborator (planner / worker / tester) is replaced with an explicit fake
before the orchestrator is constructed, so nothing here touches the network,
spawns a subprocess or reads an API key.

The module-level ``PlannerAgent`` / ``WorkerAgent`` / ``TesterAgent`` names are
patched because the orchestrator looks them up on its own module at
construction (``PlannerAgent`` in ``__init__``) and at call time (``_execute`` /
``_cycle``).
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Callable
from functools import partial

import pytest
from rich.console import Console

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import (
    MAX_EMPTY_PLAN_CYCLES,
    MAX_FAILURE_CONTEXT_LINES,
    Orchestrator,
)

RUN_TIMEOUT = 5.0

PASSED = TestResult(passed=True, summary="all green", failures=[])
FAILED = TestResult(passed=False, summary="s", failures=["boom"])


# --------------------------------------------------------------------- fakes


class FakeClient:
    """Stands in for LLMClient: the orchestrator only ever reads ``.settings``."""

    def __init__(self, *, max_cycles: int = 0, max_parallel_tasks: int = 5) -> None:
        self.settings = Settings(max_cycles=max_cycles, max_parallel_tasks=max_parallel_tasks)


class ConcurrencyTracker:
    """Records how many fake workers were inside ``run()`` at the same time."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def hold(self) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            # One loop iteration: every worker that already holds the semaphore
            # gets here before any of them is released, so the peak is
            # deterministic for a given limit.
            await asyncio.sleep(0)
        finally:
            self.active -= 1


class ScriptedPlanner:
    """Fake PlannerAgent replaying ``script`` (Plan objects or exceptions).

    The final entry repeats once the script is exhausted so ``run()`` halts via
    the stagnation counter instead of spinning forever.
    """

    def __init__(
        self,
        script: list[Plan | BaseException],
        *,
        client: object | None = None,
        calls: list[tuple[str, str | None]] | None = None,
    ) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, str | None]] = calls if calls is not None else []

    async def plan(self, goal: str, *, context: str | None = None) -> Plan:
        self.calls.append((goal, context))
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, BaseException):
            raise item
        # A real planner returns fresh models each call; deep-copying keeps the
        # per-task status mutation from leaking between cycles.
        return item.model_copy(deep=True)


class ScriptedWorker:
    """Fake WorkerAgent: ``behavior`` maps a TaskModel to its result string."""

    def __init__(
        self,
        task: TaskModel,
        *,
        client: object | None = None,
        settings: Settings | None = None,
        behavior: Callable[[TaskModel], str] | None = None,
        tracker: ConcurrencyTracker | None = None,
    ) -> None:
        self.task = task
        self.behavior = behavior or (lambda t: f"done: {t.description}")
        self.tracker = tracker

    async def run(self) -> str:
        if self.tracker is not None:
            await self.tracker.hold()
        return self.behavior(self.task)


class ScriptedTester:
    """Fake TesterAgent replaying ``script`` of TestResults (last one repeats)."""

    def __init__(
        self,
        script: list[TestResult],
        *,
        client: object | None = None,
        seen: list[tuple[str, list[str]]] | None = None,
    ) -> None:
        self.script = list(script)
        self.seen: list[tuple[str, list[str]]] = seen if seen is not None else []

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.seen.append((goal, [t.description for t in tasks]))
        return self.script[min(len(self.seen) - 1, len(self.script) - 1)]


# ------------------------------------------------------------------ builders


def make_task(task_id: str, description: str) -> TaskModel:
    return TaskModel(id=task_id, description=description)


def make_plan(*descriptions: str) -> Plan:
    return Plan(
        tasks=[make_task(str(index), text) for index, text in enumerate(descriptions, 1)],
        rationale="fake plan",
    )


def build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    planner_script: list[Plan | BaseException],
    tester_script: list[TestResult],
    goal: str = "x",
    max_cycles: int = 0,
    max_parallel_tasks: int = 5,
    worker_behavior: Callable[[TaskModel], str] | None = None,
    on_event: Callable | None = None,
    tracker: ConcurrencyTracker | None = None,
) -> tuple[Orchestrator, list[tuple[str, str | None]], list[tuple[str, list[str]]]]:
    """Patch the collaborators, then construct the orchestrator."""
    planner_calls: list[tuple[str, str | None]] = []
    tester_seen: list[tuple[str, list[str]]] = []
    module = orchestrator_module
    monkeypatch.setattr(
        module, "PlannerAgent", partial(ScriptedPlanner, planner_script, calls=planner_calls)
    )
    monkeypatch.setattr(
        module,
        "WorkerAgent",
        partial(ScriptedWorker, behavior=worker_behavior, tracker=tracker),
    )
    monkeypatch.setattr(
        module,
        "TesterAgent",
        partial(ScriptedTester, tester_script, seen=tester_seen),
    )
    client = FakeClient(max_cycles=max_cycles, max_parallel_tasks=max_parallel_tasks)
    orchestrator = Orchestrator(goal=goal, client=client, on_event=on_event)  # type: ignore[arg-type]
    return orchestrator, planner_calls, tester_seen


async def run_within(orchestrator: Orchestrator) -> None:
    """Run the loop, failing the test rather than hanging on a regression."""
    await asyncio.wait_for(orchestrator.run(), timeout=RUN_TIMEOUT)


@pytest.fixture(autouse=True)
def _quiet_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep rich output out of the test report."""
    monkeypatch.setattr(
        orchestrator_module, "console", Console(file=io.StringIO(), quiet=True, width=200)
    )


# --------------------------------------------------------------------- tests


def test_fresh_orchestrator_is_not_done() -> None:
    """Regression: an empty task list used to make ``0 >= 0`` report "done"."""
    orchestrator = Orchestrator(goal="x", client=FakeClient())

    assert orchestrator._get_tasks() == []
    assert orchestrator._is_done() is False


def test_is_done_requires_completed_tasks_and_a_passing_test() -> None:
    orchestrator = Orchestrator(goal="x", client=FakeClient())

    assert orchestrator._is_done() is False, "zero tasks must never mean done"

    first = make_task("1", "alpha")
    orchestrator._tasks.append(first)
    assert orchestrator._is_done() is False, "a pending task is not done"

    first.status = "completed"
    assert orchestrator._is_done() is False, "no test verdict yet"

    orchestrator._last_test_passed = False
    assert orchestrator._is_done() is False, "a failing test is not done"

    orchestrator._last_test_passed = True
    assert orchestrator._is_done() is True

    orchestrator._tasks.append(make_task("2", "beta"))
    orchestrator._tasks[1].status = "failed"
    assert orchestrator._is_done() is False, "one unfinished task blocks completion"

    orchestrator._tasks[1].status = "completed"
    assert orchestrator._is_done() is True


async def test_goal_survives_failing_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator, _calls, _seen = build(
        monkeypatch, planner_script=[make_plan("only task")], tester_script=[FAILED], max_cycles=4
    )

    await run_within(orchestrator)

    assert orchestrator.cycles == 4
    assert orchestrator.goal == "x", "the failure summary must never overwrite the goal"
    for goal, _context in _calls:
        assert goal == "x"


async def test_cycle_accumulates_tasks_and_skips_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, _calls, _seen = build(
        monkeypatch,
        planner_script=[make_plan("alpha", "beta"), make_plan("beta", "gamma")],
        tester_script=[FAILED],
    )

    await orchestrator._cycle()
    await orchestrator._cycle()

    recorded = orchestrator._get_tasks()
    assert [t.description for t in recorded] == ["alpha", "beta", "gamma"]
    # cycle 2 re-plans "beta" (id 1); that duplicate must not be appended again.
    assert [t.id for t in recorded] == ["1", "2", "2"]
    assert len(recorded) == len({t.description for t in recorded}), "duplicate task recorded"
    assert all(t.status == "completed" for t in recorded)
    assert len(orchestrator._seen) == 3
    assert orchestrator.cycles == 0, "_cycle() does not touch the cycle counter"


async def test_get_tasks_returns_a_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator, _calls, _seen = build(
        monkeypatch, planner_script=[make_plan("alpha")], tester_script=[FAILED]
    )

    await orchestrator._cycle()
    snapshot = orchestrator._get_tasks()
    snapshot.clear()

    assert len(orchestrator._get_tasks()) == 1, "_get_tasks() must not expose the live list"


async def test_max_cycles_halts_after_exactly_one_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, _calls, _seen = build(
        monkeypatch,
        planner_script=[make_plan("alpha"), make_plan("beta")],
        tester_script=[FAILED],
        max_cycles=1,
    )

    await run_within(orchestrator)

    assert orchestrator.cycles == 1


async def test_parallel_execution_respects_the_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = ConcurrencyTracker()
    orchestrator, _calls, _seen = build(
        monkeypatch,
        planner_script=[make_plan("a", "b", "c", "d")],
        tester_script=[FAILED],
        max_parallel_tasks=2,
        tracker=tracker,
    )

    await orchestrator._cycle()

    assert len(orchestrator._get_tasks()) == 4
    assert tracker.peak <= 2, f"ran {tracker.peak} tasks in parallel with a limit of 2"
    assert tracker.peak >= 2, "the limit of 2 should actually be reached"
    assert tracker.active == 0


async def test_repeated_plan_halts_via_the_stagnation_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, _calls, _seen = build(
        monkeypatch, planner_script=[make_plan("always the same")], tester_script=[FAILED]
    )

    await run_within(orchestrator)

    assert orchestrator.cycles <= MAX_EMPTY_PLAN_CYCLES + 2
    assert orchestrator._empty_plan_cycles >= MAX_EMPTY_PLAN_CYCLES


async def test_stop_before_run_exits_without_cycling(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    orchestrator, _calls, _seen = build(
        monkeypatch,
        planner_script=[make_plan("alpha")],
        tester_script=[FAILED],
        on_event=events.append,
    )

    orchestrator.stop()
    await run_within(orchestrator)

    assert orchestrator.cycles == 0
    assert orchestrator._get_tasks() == []
    assert "stop requested, halting" in events


async def test_stop_from_an_event_callback_halts_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    orchestrator, _calls, _seen = build(
        monkeypatch,
        planner_script=[make_plan("alpha"), make_plan("beta")],
        tester_script=[FAILED],
        max_cycles=10,
        on_event=events.append,
    )
    original_emit = orchestrator._emit

    async def emit_and_stop(message: str) -> None:
        await original_emit(message)
        if message.startswith("cycle 1 starting"):
            orchestrator.stop()

    orchestrator._emit = emit_and_stop  # type: ignore[method-assign]

    await run_within(orchestrator)

    assert orchestrator.cycles == 1
    assert orchestrator._stop_requested is True
    assert "stop requested, halting" in events


async def test_sync_event_collector_receives_the_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    orchestrator, _calls, _seen = build(
        monkeypatch,
        planner_script=[make_plan("alpha")],
        tester_script=[FAILED],
        max_cycles=1,
        on_event=events.append,
    )

    await run_within(orchestrator)

    assert any(e == "cycle 1 starting" for e in events)
    assert any(e == "plan produced with 1 task(s)" for e in events)
    assert any(e == "task 1 started: alpha" for e in events)
    assert any(e == "task 1 completed" for e in events)
    assert any(e == "tests failed: s" for e in events)
    assert any("max cycles (1) reached" in e for e in events)


async def test_async_event_collector_is_awaited_and_delivers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    in_flight: list[int] = []
    finished: list[int] = []

    async def collector(message: str) -> None:
        in_flight.append(1)
        await asyncio.sleep(0)
        events.append(message)
        finished.append(1)
        in_flight.clear()

    orchestrator, _calls, _seen = build(
        monkeypatch,
        planner_script=[make_plan("alpha")],
        tester_script=[FAILED],
        max_cycles=1,
        on_event=collector,
    )

    await run_within(orchestrator)

    # Every emit awaited the coroutine to completion before returning, so the
    # two lists stay in lockstep and nothing is left in flight.
    assert len(events) == len(finished) >= 6
    assert in_flight == []
    assert any(e == "task 1 completed" for e in events)


async def test_raising_event_collector_does_not_crash_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def collector(message: str) -> None:
        raise RuntimeError(f"consumer is gone: {message}")

    orchestrator, _calls, _seen = build(
        monkeypatch,
        planner_script=[make_plan("alpha"), make_plan("beta")],
        tester_script=[FAILED],
        max_cycles=2,
        on_event=collector,
    )

    await run_within(orchestrator)

    assert orchestrator.cycles == 2
    assert [t.description for t in orchestrator._get_tasks()] == ["alpha", "beta"]


async def test_planner_receives_the_previous_failure_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, calls, _seen = build(
        monkeypatch,
        planner_script=[make_plan("alpha"), make_plan("beta")],
        tester_script=[FAILED],
        max_cycles=2,
    )

    await run_within(orchestrator)

    assert calls[0][1] == "", "the first cycle has no failure context"
    assert calls[1][1] is not None
    assert "Fix failing tests: s" in calls[1][1]
    assert "- boom" in calls[1][1]


async def test_failure_context_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    failures = [f"failure-{index}" for index in range(40)]
    orchestrator, calls, _seen = build(
        monkeypatch,
        planner_script=[make_plan("alpha"), make_plan("beta")],
        tester_script=[TestResult(passed=False, summary="many", failures=failures)],
        max_cycles=2,
    )

    await run_within(orchestrator)

    lines = (calls[1][1] or "").splitlines()
    assert len(lines) == MAX_FAILURE_CONTEXT_LINES
    assert lines[-1] == "- failure-39"


async def test_planner_exception_does_not_escape_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    orchestrator, _calls, _seen = build(
        monkeypatch,
        planner_script=[make_plan("alpha"), RuntimeError("planner exploded")],
        tester_script=[FAILED],
        on_event=events.append,
    )

    await run_within(orchestrator)

    assert any("planner failed: RuntimeError: planner exploded" in e for e in events)
    assert orchestrator.cycles <= MAX_EMPTY_PLAN_CYCLES + 2
    assert orchestrator._empty_plan_cycles >= MAX_EMPTY_PLAN_CYCLES


async def test_empty_plan_halts_via_the_stagnation_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, _calls, seen = build(
        monkeypatch,
        planner_script=[Plan(tasks=[], rationale="nothing to do")],
        tester_script=[FAILED],
    )

    await run_within(orchestrator)

    assert orchestrator.cycles == MAX_EMPTY_PLAN_CYCLES
    assert orchestrator._empty_plan_cycles >= MAX_EMPTY_PLAN_CYCLES
    assert seen == [], "an empty plan must not run the tester"


async def test_planner_that_always_fails_terminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, _calls, _seen = build(
        monkeypatch,
        planner_script=[ValueError("no plan for you")],
        tester_script=[FAILED],
    )

    await run_within(orchestrator)

    assert orchestrator.cycles == MAX_EMPTY_PLAN_CYCLES
    assert orchestrator.plan is None


async def test_worker_failure_is_recorded_and_blocks_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def behavior(task: TaskModel) -> str:
        if task.description == "bad":
            raise RuntimeError("worker exploded")
        return f"done: {task.description}"

    events: list[str] = []
    orchestrator, _calls, _seen = build(
        monkeypatch,
        planner_script=[make_plan("good", "bad")],
        tester_script=[FAILED],
        worker_behavior=behavior,
        max_cycles=1,
        on_event=events.append,
    )

    await run_within(orchestrator)

    by_id = {t.description: t for t in orchestrator._get_tasks()}
    assert by_id["good"].status == "completed"
    assert by_id["bad"].status == "failed"
    assert "worker exploded" in (by_id["bad"].result or "")
    assert orchestrator._is_done() is False
    assert any("task 2 failed: worker exploded" in e for e in events)
