"""Loop-control tests for :class:`furrow.core.orchestrator.Orchestrator`.

Everything else in Furrow is a worker that the orchestrator's loop drives, so
the loop is the product: these tests pin when it stops, what it feeds the next
cycle, how it orders tasks, and what it tells the outside world.

How the loop is driven
----------------------
The three agent classes the orchestrator imports at module scope are replaced
per test (:func:`install_fake_agents`) with tiny fakes that record every call and
replay a script. Nothing here reaches a provider: plans come from a list (or a
``plan_for`` callable, so each cycle can get a *fresh* plan the way a real
re-planning run does), verdicts from a list or ``verdict_for``, and the worker is
a coroutine that records start/end ordering and can overlap on demand.

The fakes are created per test inside a closure over one :class:`Script` object,
so no state can leak between tests -- ``pytest``'s ``monkeypatch`` restores the
real agent classes afterwards.

Private access
--------------
The retry-budget tests reach ``Orchestrator._execute_waves()`` through the
:func:`dispatch` helper, as does the stalled-wave guard test. The loop re-plans
every cycle, so a task object is only ever dispatched once and ``_ready_tasks``
only hands out ``PENDING`` tasks; the ``max_task_attempts`` escalation is
therefore unreachable from the outside. Everything else goes through the public
``run()`` / ``stop()`` / ``is_running`` API and the event stream.

Two plan shapes used to kill the loop outright and are now pinned as working
behaviour, because both were live production bugs rather than edge cases:

* a dependency cycle *plus* a task that depends on a member of it used to make
  ``_detect_cycles`` raise ``ValueError`` out of ``run()``, killing the process
  with no ``error`` and no ``done`` event;
* requesting ``stop()`` while a wave with several ready tasks was still
  dispatching used to spin in ``_execute_waves`` forever, so ``run()`` never
  returned.

Both tests that exercise them are wrapped in ``asyncio.wait_for`` so a
regression fails the test instead of hanging the suite.

One test cannot be wrapped that way. Whether the cycle loop ever hands control
back to the event loop is the one property that ``wait_for`` is unable to
observe: a loop that never yields starves every coroutine on its own loop,
including the ``wait_for`` that was supposed to stop it. It is therefore pinned
by running a self-stopping orchestrator in a subprocess and giving the child a
deadline -- see ``SPINNING_PROGRAM``.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import furrow.core.orchestrator as orchestrator_module
from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult
from furrow.core.orchestrator import (
    STOPPED_BEFORE_START,
    SUCCESS_REASON,
    Orchestrator,
    PlannerUnavailableError,
)
from furrow.core.state import StateStore
from furrow.llm import LLMClient
from tests.conftest import (
    SettingsFactory,
    StubLLMClient,
    make_plan,
    make_task,
)

DEFAULT_GOAL = "ship the parser"

#: Payload keys the orchestrator documents for every event it emits.
EVENT_KEYS: dict[str, set[str]] = {
    "cycle_start": {"cycle"},
    "plan": {"rationale", "task_count", "tasks"},
    "task_start": {"id", "description"},
    "task_done": {"id", "status", "result", "error"},
    "test_result": {"passed", "summary", "failures"},
    "cycle_end": {"cycle", "passed", "completed", "failed"},
    "done": {"reason", "cycles"},
    "error": {"message", "where"},
}

Event = tuple[str, dict[str, Any]]
PlanMaker = Callable[[int], Plan]
VerdictMaker = Callable[[int], TestResult]
WorkerHook = Callable[[TaskModel], Awaitable[None]]
EventHook = Callable[[str, dict[str, Any]], None]
RigBuilder = Callable[..., "Rig"]


class ScriptExhaustedError(BaseException):
    """Raised when a test drives more cycles than it scripted.

    Deliberately a :class:`BaseException`: the orchestrator swallows
    ``Exception`` around the planner and the tester, so an ordinary
    ``AssertionError`` here would be downgraded to an ``error`` event and the
    test would pass for the wrong reason.
    """


def _noop() -> None:
    """Default no-op for the script's ``request_stop`` hook."""


def _not_running() -> bool:
    """Default for the script's ``is_running`` probe before a rig is attached."""
    return False


@dataclass(frozen=True, slots=True)
class PlanCall:
    """One recorded ``PlannerAgent.plan`` invocation."""

    goal: str
    context: str


@dataclass(frozen=True, slots=True)
class VerdictCall:
    """One recorded ``TesterAgent.run`` invocation."""

    goal: str
    task_ids: tuple[str, ...]


@dataclass
class Script:
    """Everything the fake agents record, plus what they should reply."""

    plans: list[Plan | BaseException] = field(default_factory=list)
    verdicts: list[TestResult | BaseException] = field(default_factory=list)
    plan_for: PlanMaker | None = None
    verdict_for: VerdictMaker | None = None
    worker_hook: WorkerHook | None = None
    planner_calls: list[PlanCall] = field(default_factory=list)
    tester_calls: list[VerdictCall] = field(default_factory=list)
    started: list[str] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)
    order: list[tuple[str, str]] = field(default_factory=list)
    start_snapshots: dict[str, tuple[str, ...]] = field(default_factory=dict)
    worker_started: asyncio.Event = field(default_factory=asyncio.Event)
    block_tasks: asyncio.Event | None = None
    live: int = 0
    peak: int = 0
    stop_on_plan: int | None = None
    stop_on_test: int | None = None
    request_stop: Callable[[], None] = _noop
    is_running: Callable[[], bool] = _not_running

    def attach(self, orchestrator: Orchestrator) -> None:
        """Let the fakes reach the orchestrator they were installed for."""
        self.request_stop = orchestrator.stop
        self.is_running = lambda: orchestrator.is_running

    def next_plan(self, cycle: int) -> Plan:
        if self.plans:
            item = self.plans.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        if self.plan_for is not None:
            return self.plan_for(cycle)
        raise ScriptExhaustedError(f"no plan scripted for cycle {cycle}")

    def next_verdict(self, cycle: int) -> TestResult:
        if self.verdicts:
            item = self.verdicts.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        if self.verdict_for is not None:
            return self.verdict_for(cycle)
        raise ScriptExhaustedError(f"no verdict scripted for cycle {cycle}")


@dataclass
class Rig:
    """An orchestrator plus everything its fake agents recorded."""

    orchestrator: Orchestrator
    script: Script
    settings: Settings
    events: list[Event]


def install_fake_agents(monkeypatch: pytest.MonkeyPatch, script: Script) -> None:
    """Replace the agent classes as ``furrow.core.orchestrator`` sees them.

    The orchestrator binds ``PlannerAgent`` / ``WorkerAgent`` / ``TesterAgent``
    at import time, so patching the module attributes is what makes
    ``Orchestrator(...)`` build fakes (and ``_make_worker`` hand one out).

    The worker fake mirrors the real ``WorkerAgent.run`` bookkeeping that the
    retry budget depends on: it counts the attempt before doing any work and
    re-raises, leaving the status escalation to the orchestrator.
    """

    class FakePlanner:
        def __init__(
            self,
            client: LLMClient | None = None,
            settings: Settings | None = None,
        ) -> None:
            self.client = client
            self.settings = settings

        async def plan(self, goal: str, context: str = "") -> Plan:
            script.planner_calls.append(PlanCall(goal=goal, context=context))
            cycle = len(script.planner_calls)
            if script.stop_on_plan is not None and cycle >= script.stop_on_plan:
                script.request_stop()
            return script.next_plan(cycle)

    class FakeWorker:
        def __init__(
            self,
            task: TaskModel,
            client: LLMClient | None = None,
            settings: Settings | None = None,
        ) -> None:
            self.task = task
            self.client = client
            self.settings = settings

        async def run(self) -> str:
            task = self.task
            task.attempts += 1
            task.status = TaskStatus.RUNNING
            task.error = None
            script.started.append(task.id)
            script.start_snapshots[task.id] = tuple(script.finished)
            script.order.append(("start", task.id))
            script.live += 1
            script.peak = max(script.peak, script.live)
            script.worker_started.set()
            try:
                if script.block_tasks is not None:
                    await script.block_tasks.wait()
                elif script.worker_hook is None:
                    await asyncio.sleep(0)
                else:
                    await script.worker_hook(task)
            except Exception as exc:
                script.live -= 1
                script.order.append(("failed", task.id))
                task.status = TaskStatus.FAILED
                task.error = f"{type(exc).__name__}: {exc}"
                raise
            script.live -= 1
            script.finished.append(task.id)
            script.order.append(("end", task.id))
            task.status = TaskStatus.COMPLETED
            task.result = f"completed {task.id}"
            return task.result

    class FakeTester:
        def __init__(
            self,
            client: LLMClient | None = None,
            settings: Settings | None = None,
        ) -> None:
            self.client = client
            self.settings = settings

        async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
            script.tester_calls.append(
                VerdictCall(goal=goal, task_ids=tuple(task.id for task in tasks))
            )
            cycle = len(script.tester_calls)
            if script.stop_on_test is not None and cycle >= script.stop_on_test:
                script.request_stop()
            return script.next_verdict(cycle)

    monkeypatch.setattr(orchestrator_module, "PlannerAgent", FakePlanner)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", FakeTester)


@pytest.fixture
def make_rig(
    monkeypatch: pytest.MonkeyPatch,
    settings_factory: SettingsFactory,
) -> RigBuilder:
    """Build an :class:`Orchestrator` wired to scripted fake agents."""

    def build(
        *,
        plans: Sequence[Plan | BaseException] = (),
        plan_for: PlanMaker | None = None,
        verdicts: Sequence[TestResult | BaseException] = (),
        verdict_for: VerdictMaker | None = None,
        worker_hook: WorkerHook | None = None,
        settings_overrides: dict[str, Any] | None = None,
        goal: str = DEFAULT_GOAL,
        on_event: EventHook | None = None,
    ) -> Rig:
        script = Script(
            plans=list(plans),
            plan_for=plan_for,
            verdicts=list(verdicts),
            verdict_for=verdict_for,
            worker_hook=worker_hook,
        )
        install_fake_agents(monkeypatch, script)
        settings = settings_factory(**(settings_overrides or {}))
        events: list[Event] = []

        def record(name: str, payload: dict[str, Any]) -> None:
            events.append((name, dict(payload)))
            if on_event is not None:
                on_event(name, payload)

        orchestrator = Orchestrator(
            goal,
            client=StubLLMClient(settings=settings),
            settings=settings,
            on_event=record,
        )
        script.attach(orchestrator)
        return Rig(orchestrator=orchestrator, script=script, settings=settings, events=events)

    return build


# --------------------------------------------------------------------------- #
# Verdict / plan shorthands
# --------------------------------------------------------------------------- #


def passing(summary: str = "all checks green") -> TestResult:
    """A green tester verdict."""
    return TestResult(passed=True, summary=summary)


def failing(*failures: str, summary: str = "tests failed") -> TestResult:
    """A red tester verdict carrying the given failure lines."""
    return TestResult(passed=False, summary=summary, failures=list(failures))


def one_task_plan(task_id: str = "t1", rationale: str = "one task") -> Plan:
    """A single-task plan that completes cleanly."""
    return make_plan([make_task(task_id, f"do {task_id}")], rationale)


def failing_plan(task_id: str = "t1") -> Plan:
    """A single-task plan to pair with a worker hook that fails every task."""
    return make_plan([make_task(task_id, f"do {task_id}")], "retry this")


def fan_out_plan() -> Plan:
    """``b`` and ``c`` both depend on ``a``: one task in wave one, two in wave two."""
    return make_plan(
        [
            make_task("a", "write the parser"),
            make_task("b", "wire it in", dependencies=["a"]),
            make_task("c", "document it", dependencies=["a"]),
        ],
        "parser work",
    )


async def always_fails(task: TaskModel) -> None:
    """Worker hook that fails the task the way a broken LLM call would."""
    raise RuntimeError("boom")


def barrier_hook(needed: int) -> WorkerHook:
    """A worker hook that only lets tasks finish once ``needed`` of them overlap.

    With the concurrency cap in place exactly ``needed`` tasks can be inside the
    hook at once, so they release each other. If the cap were removed the hook
    would see all of them at once instead, which is what ``Script.peak`` records.
    The ``wait_for`` is a liveness backstop only: it turns a hypothetical hang
    into a failed task rather than a stuck test.
    """
    arrivals: set[str] = set()
    released = asyncio.Event()

    async def hook(task: TaskModel) -> None:
        arrivals.add(task.id)
        if len(arrivals) >= needed:
            released.set()
        await asyncio.wait_for(released.wait(), timeout=1)

    return hook


async def dispatch(rig: Rig, task: TaskModel) -> None:
    """Execute one wave whose only member is ``task``.

    ``Orchestrator._execute_waves`` is called directly on purpose: the loop
    re-plans every cycle, so a task object is only ever dispatched once and the
    ``max_task_attempts`` escalation cannot be reached through ``run()``.
    Handing the same object back means marking it ``PENDING`` again first, which
    is what a within-cycle retry would look like.
    """
    rig.orchestrator.tasks = [task]
    await rig.orchestrator._execute_waves()


async def run_until_cancelled(rig: Rig) -> None:
    """Run until a task is executing, cancel the run, then await the cancellation."""
    runner = asyncio.create_task(rig.orchestrator.run())
    await asyncio.wait_for(rig.script.worker_started.wait(), timeout=1)
    runner.cancel()
    await runner


# --------------------------------------------------------------------------- #
# Event-log helpers
# --------------------------------------------------------------------------- #


def event_names(events: Sequence[Event]) -> list[str]:
    """Every emitted event name, in order."""
    return [name for name, _ in events]


def payloads(events: Sequence[Event], name: str) -> list[dict[str, Any]]:
    """Every payload emitted under ``name``, in order."""
    return [payload for event, payload in events if event == name]


def only_payload(events: Sequence[Event], name: str) -> dict[str, Any]:
    """The single payload emitted under ``name``."""
    found = payloads(events, name)
    assert len(found) == 1, f"expected exactly one {name!r} event, got {len(found)}"
    return found[0]


def event_index(events: Sequence[Event], name: str, task_id: str) -> int:
    """Position of the ``name`` event for ``task_id`` in the event log."""
    for index, (event, payload) in enumerate(events):
        if event == name and payload.get("id") == task_id:
            return index
    raise AssertionError(f"no {name!r} event was emitted for task {task_id!r}")


def state_path(settings: Settings) -> Path:
    """The configured state file, asserting the fixture always set one."""
    path = settings.state_file
    assert path is not None, "settings_factory always configures a state_file"
    return path


# --------------------------------------------------------------------------- #
# 1. Completion detection
# --------------------------------------------------------------------------- #


async def test_run_halts_after_one_cycle_when_tasks_complete_and_tests_pass(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(plans=[one_task_plan()], verdicts=[passing()])
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 1
    assert only_payload(rig.events, "done") == {
        "reason": "All tasks completed and tests passed",
        "cycles": 1,
    }


async def test_all_tasks_are_marked_completed_on_the_happy_path(make_rig: RigBuilder) -> None:
    rig = make_rig(plans=[one_task_plan("t1")], verdicts=[passing()])
    await rig.orchestrator.run()
    assert [task.status for task in rig.orchestrator.tasks] == [TaskStatus.COMPLETED]


async def test_a_failed_task_keeps_the_loop_running(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plan_for=lambda cycle: failing_plan(f"t{cycle}"),
        verdict_for=lambda cycle: passing(),
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 2},
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 2
    assert only_payload(rig.events, "done")["reason"] == "Reached max_cycles (2)"


async def test_failing_tests_keep_the_loop_running_even_when_every_task_completed(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plan_for=lambda cycle: one_task_plan(f"t{cycle}"),
        verdict_for=lambda cycle: failing(f"failure-{cycle}"),
        settings_overrides={"max_cycles": 2},
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 2
    assert only_payload(rig.events, "done")["reason"] == "Reached max_cycles (2)"


async def test_a_blocked_task_keeps_the_loop_running(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plan_for=lambda cycle: make_plan(
            [make_task("t1", "orphan", dependencies=["ghost"])], "dangling dep"
        ),
        verdict_for=lambda cycle: passing(),
        settings_overrides={"max_cycles": 2},
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 2
    assert rig.orchestrator.tasks[0].status is TaskStatus.BLOCKED


async def test_an_empty_plan_with_no_failures_is_done(make_rig: RigBuilder) -> None:
    rig = make_rig(plans=[make_plan([], "nothing to do")], verdicts=[passing()])
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 1
    assert only_payload(rig.events, "done")["cycles"] == 1


async def test_a_genuinely_empty_plan_still_reports_the_success_reason(
    make_rig: RigBuilder,
) -> None:
    """The other half of Defect 1: a real "nothing to do" is still a success.

    Pinned separately from the test above because it is the test that catches the
    over-correction -- making the loop unable to declare victory would be just as
    wrong as letting it declare victory for a run that achieved nothing.
    """
    rig = make_rig(plans=[make_plan([], "nothing to do")], verdicts=[passing()])
    await rig.orchestrator.run()
    assert only_payload(rig.events, "done") == {"reason": SUCCESS_REASON, "cycles": 1}


async def test_a_genuinely_empty_plan_after_a_planner_failure_still_reports_success(
    make_rig: RigBuilder,
) -> None:
    """Recovery means recovery: the planner saying "nothing to do" is still a plan."""
    rig = make_rig(
        plans=[RuntimeError("planner exploded"), make_plan([], "nothing to do")],
        verdicts=[passing(), passing()],
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 2
    assert only_payload(rig.events, "done") == {"reason": SUCCESS_REASON, "cycles": 2}


# --------------------------------------------------------------------------- #
# 2. Goal preservation
# --------------------------------------------------------------------------- #


async def test_goal_survives_a_cycle_whose_tests_failed(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plans=[failing_plan()],
        verdicts=[failing("assert 1 == 2")],
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.goal == DEFAULT_GOAL


async def test_the_failure_text_reaches_the_planner_as_context(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plans=[one_task_plan("t1"), one_task_plan("t2")],
        verdicts=[failing("assert 1 == 2", "assert 2 == 3"), passing()],
    )
    await rig.orchestrator.run()
    assert rig.script.planner_calls[1].context == "assert 1 == 2\nassert 2 == 3"


async def test_the_planner_keeps_receiving_the_original_goal_on_every_cycle(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[one_task_plan("t1"), one_task_plan("t2")],
        verdicts=[failing("boom"), passing()],
    )
    await rig.orchestrator.run()
    assert [call.goal for call in rig.script.planner_calls] == [DEFAULT_GOAL, DEFAULT_GOAL]


async def test_the_first_cycle_hands_the_planner_the_goal_as_its_context(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(plans=[one_task_plan()], verdicts=[passing()])
    await rig.orchestrator.run()
    assert rig.script.planner_calls[0].context == DEFAULT_GOAL


async def test_the_tester_is_always_given_the_original_goal(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plans=[one_task_plan("t1"), one_task_plan("t2")],
        verdicts=[failing("boom"), passing()],
    )
    await rig.orchestrator.run()
    assert [call.goal for call in rig.script.tester_calls] == [DEFAULT_GOAL, DEFAULT_GOAL]


async def test_the_focus_carries_the_failure_text_into_the_next_cycle(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[one_task_plan("t1"), one_task_plan("t2")],
        verdicts=[failing("boom", "bang"), passing()],
    )
    await rig.orchestrator.run()
    assert rig.script.planner_calls[1].context == "boom\nbang"


async def test_a_passing_cycle_leaves_the_focus_on_the_last_failing_cycle(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[one_task_plan("t1"), one_task_plan("t2")],
        verdicts=[failing("boom", "bang"), passing()],
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.focus == "boom\nbang"


# --------------------------------------------------------------------------- #
# 3. Dependency-aware waves
# --------------------------------------------------------------------------- #


async def test_dependency_waves_do_not_start_dependents_early(make_rig: RigBuilder) -> None:
    rig = make_rig(plans=[fan_out_plan()], verdicts=[passing()])
    await rig.orchestrator.run()
    assert rig.script.started == ["a", "b", "c"]
    assert event_index(rig.events, "task_start", "a") < event_index(
        rig.events, "task_start", "b"
    )
    assert event_index(rig.events, "task_done", "a") < event_index(
        rig.events, "task_start", "b"
    )
    assert event_index(rig.events, "task_done", "a") < event_index(
        rig.events, "task_start", "c"
    )


async def test_a_task_with_no_dependencies_runs_alone_in_the_first_wave(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(plans=[fan_out_plan()], verdicts=[passing()])
    await rig.orchestrator.run()
    assert rig.script.start_snapshots == {"a": (), "b": ("a",), "c": ("a",)}


async def test_dependents_share_a_wave_once_their_prerequisite_is_done(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(plans=[fan_out_plan()], verdicts=[passing()])
    await rig.orchestrator.run()
    assert rig.script.order[:4] == [
        ("start", "a"),
        ("end", "a"),
        ("start", "b"),
        ("start", "c"),
    ]


async def test_a_three_level_dependency_chain_runs_one_task_per_wave(
    make_rig: RigBuilder,
) -> None:
    plan = make_plan(
        [
            make_task("a", "write it"),
            make_task("b", "test it", dependencies=["a"]),
            make_task("c", "ship it", dependencies=["b"]),
        ],
        "sequential",
    )
    rig = make_rig(plans=[plan], verdicts=[passing()])
    await rig.orchestrator.run()
    assert rig.script.start_snapshots == {"a": (), "b": ("a",), "c": ("a", "b")}


async def test_a_task_whose_dependency_failed_never_starts(make_rig: RigBuilder) -> None:
    plan = make_plan(
        [
            make_task("a", "write it"),
            make_task("b", "test it", dependencies=["a"]),
        ],
        "sequential",
    )
    rig = make_rig(
        plans=[plan],
        verdicts=[passing()],
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    assert rig.script.started == ["a"]


async def test_a_task_whose_dependency_failed_is_blocked_not_stranded(
    make_rig: RigBuilder,
) -> None:
    # This used to assert ``PENDING``. The old expectation encoded Defect 2: the
    # task was never ready (its dependency did not complete) and never blocked
    # (the dependency did not end BLOCKED), so it left the run pending with
    # ``error=None``, no ``task_done`` event and nothing to explain it.
    plan = make_plan(
        [
            make_task("a", "write it"),
            make_task("b", "test it", dependencies=["a"]),
        ],
        "sequential",
    )
    rig = make_rig(
        plans=[plan],
        verdicts=[passing()],
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.tasks[1].status is TaskStatus.BLOCKED


# --------------------------------------------------------------------------- #
# 4. Blocked detection
# --------------------------------------------------------------------------- #


async def test_a_task_depending_on_an_unknown_id_is_blocked(make_rig: RigBuilder) -> None:
    plan = make_plan([make_task("t1", "orphan", dependencies=["ghost"])], "dangling")
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await rig.orchestrator.run()
    assert rig.orchestrator.tasks[0].status is TaskStatus.BLOCKED


async def test_a_blocked_task_explains_the_missing_dependency(make_rig: RigBuilder) -> None:
    plan = make_plan([make_task("t1", "orphan", dependencies=["ghost"])], "dangling")
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await rig.orchestrator.run()
    assert "'ghost' does not exist in plan" in (rig.orchestrator.tasks[0].error or "")


async def test_a_self_dependent_task_is_blocked(make_rig: RigBuilder) -> None:
    plan = make_plan([make_task("t1", "loop", dependencies=["t1"])], "self")
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await rig.orchestrator.run()
    assert rig.orchestrator.tasks[0].status is TaskStatus.BLOCKED


async def test_a_self_dependent_task_explains_the_self_dependency(make_rig: RigBuilder) -> None:
    plan = make_plan([make_task("t1", "loop", dependencies=["t1"])], "self")
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await rig.orchestrator.run()
    assert rig.orchestrator.tasks[0].error == "Self-dependency detected"


async def test_a_self_dependent_task_never_runs(make_rig: RigBuilder) -> None:
    plan = make_plan([make_task("t1", "loop", dependencies=["t1"])], "self")
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await rig.orchestrator.run()
    assert rig.script.started == []


async def test_a_two_node_dependency_cycle_blocks_both_tasks(make_rig: RigBuilder) -> None:
    plan = make_plan(
        [
            make_task("a", "needs b", dependencies=["b"]),
            make_task("b", "needs a", dependencies=["a"]),
        ],
        "cycle",
    )
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await rig.orchestrator.run()
    assert [task.status for task in rig.orchestrator.tasks] == [
        TaskStatus.BLOCKED,
        TaskStatus.BLOCKED,
    ]


async def test_a_dependency_cycle_is_reported_as_a_cycle(make_rig: RigBuilder) -> None:
    plan = make_plan(
        [
            make_task("a", "needs b", dependencies=["b"]),
            make_task("b", "needs a", dependencies=["a"]),
        ],
        "cycle",
    )
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await rig.orchestrator.run()
    assert rig.orchestrator.tasks[0].error == "Dependency cycle detected"


async def test_a_longer_dependency_cycle_blocks_every_task_in_it(make_rig: RigBuilder) -> None:
    plan = make_plan(
        [
            make_task("a", "needs b", dependencies=["b"]),
            make_task("b", "needs c", dependencies=["c"]),
            make_task("c", "needs a", dependencies=["a"]),
        ],
        "cycle",
    )
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await rig.orchestrator.run()
    assert [task.status for task in rig.orchestrator.tasks] == [
        TaskStatus.BLOCKED,
        TaskStatus.BLOCKED,
        TaskStatus.BLOCKED,
    ]


async def test_a_task_whose_dependency_is_blocked_is_blocked_too(
    make_rig: RigBuilder,
) -> None:
    plan = make_plan(
        [
            make_task("a", "needs a ghost", dependencies=["ghost"]),
            make_task("b", "needs a", dependencies=["a"]),
            make_task("c", "needs b", dependencies=["b"]),
        ],
        "chain of blocked work",
    )
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await rig.orchestrator.run()
    assert [task.status for task in rig.orchestrator.tasks] == [
        TaskStatus.BLOCKED,
        TaskStatus.BLOCKED,
        TaskStatus.BLOCKED,
    ]


async def test_a_transitively_blocked_task_explains_the_blocked_dependency(
    make_rig: RigBuilder,
) -> None:
    plan = make_plan(
        [
            make_task("a", "needs a ghost", dependencies=["ghost"]),
            make_task("b", "needs a", dependencies=["a"]),
        ],
        "chain of blocked work",
    )
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await rig.orchestrator.run()
    assert "Dependency 'a' is blocked" in (rig.orchestrator.tasks[1].error or "")


def dependent_on_failed_plan() -> Plan:
    """``a`` has no dependencies; ``b`` cannot run unless ``a`` completes."""
    return make_plan(
        [
            make_task("a", "write it"),
            make_task("b", "test it", dependencies=["a"]),
        ],
        "sequential",
    )


async def test_a_task_stranded_by_a_failed_dependency_explains_it(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[dependent_on_failed_plan()],
        verdicts=[passing()],
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.tasks[1].error == (
        "Dependency 'a' ended failed and can never complete"
    )


async def test_a_task_stranded_by_a_failed_dependency_emits_task_done(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[dependent_on_failed_plan()],
        verdicts=[passing()],
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    assert [(payload["id"], payload["status"]) for payload in payloads(rig.events, "task_done")] == [
        ("a", TaskStatus.FAILED),
        ("b", TaskStatus.BLOCKED),
    ]


async def test_a_task_whose_dependency_was_skipped_is_blocked(make_rig: RigBuilder) -> None:
    """``skipped`` is terminal too, so it strands dependants exactly like ``failed``."""
    rig = make_rig(
        plans=[dependent_on_failed_plan()],
        verdicts=[passing()],
        worker_hook=always_fails,
        # One attempt is one chance: the failure escalates a straight to SKIPPED.
        settings_overrides={"max_task_attempts": 1, "max_cycles": 1},
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.tasks[1].error == (
        "Dependency 'a' ended skipped and can never complete"
    )


async def test_a_dependant_listed_before_the_task_that_stranded_it_is_blocked_too(
    make_rig: RigBuilder,
) -> None:
    """The close-out runs to a fixpoint, so plan order cannot hide a stranded task."""
    plan = make_plan(
        [
            make_task("c", "after b", dependencies=["b"]),
            make_task("b", "after a", dependencies=["a"]),
            make_task("a", "write it"),
        ],
        "reverse-ordered chain",
    )
    rig = make_rig(
        plans=[plan],
        verdicts=[passing()],
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    assert [(task.id, task.status) for task in rig.orchestrator.tasks] == [
        ("c", TaskStatus.BLOCKED),
        ("b", TaskStatus.BLOCKED),
        ("a", TaskStatus.FAILED),
    ]


async def test_a_dependant_of_a_stranded_task_reports_the_blocked_wording(
    make_rig: RigBuilder,
) -> None:
    """Once stranded work is BLOCKED, its own dependants follow ``_propagate_blocked``."""
    plan = make_plan(
        [
            make_task("a", "write it"),
            make_task("b", "after a", dependencies=["a"]),
            make_task("c", "after b", dependencies=["b"]),
        ],
        "chain below a failure",
    )
    rig = make_rig(
        plans=[plan],
        verdicts=[passing()],
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.tasks[2].error == "Dependency 'b' is blocked"


async def test_no_run_ever_ends_with_a_task_pending_without_an_explanation(
    make_rig: RigBuilder,
) -> None:
    """The invariant behind Defect 2, asserted across every plan shape at once.

    Whatever a plan looks like and however its tasks die, a run may not leave a
    task ``PENDING``: ``PENDING`` means "waiting to be dispatched", and the wave
    loop has finished by the time the state is written, so a pending task at
    that point is a task nobody is ever going to look at again.
    """
    shapes: dict[str, Plan] = {
        "one task": make_plan([make_task("a", "write it")], ""),
        "dependent on a failure": dependent_on_failed_plan(),
        "wave": fan_out_plan(),
        "chain": make_plan(
            [
                make_task("a", "root"),
                make_task("b", "after a", dependencies=["a"]),
                make_task("c", "after b", dependencies=["b"]),
            ],
            "chain",
        ),
        "reverse-ordered chain": make_plan(
            [
                make_task("c", "after b", dependencies=["b"]),
                make_task("b", "after a", dependencies=["a"]),
                make_task("a", "write it"),
            ],
            "reverse-ordered chain",
        ),
        "dangling dependency": make_plan([make_task("a", "orphan", dependencies=["ghost"])], ""),
        "self dependency": make_plan([make_task("a", "loop", dependencies=["a"])], ""),
        "cycle with a dependant": cycle_plus_dependent_plan(),
    }
    for name, plan in shapes.items():
        rig = make_rig(
            plans=[plan],
            verdicts=[failing("boom")],
            worker_hook=always_fails,
            # One attempt per task, so failures escalate to SKIPPED and the
            # shapes cover both dead-end statuses as well as BLOCKED.
            settings_overrides={"max_task_attempts": 1, "max_cycles": 1},
        )
        await rig.orchestrator.run()
        assert [
            (task.id, task.status) for task in rig.orchestrator.tasks if not task.is_terminal()
        ] == [], f"{name}: a task was left pending"
        assert [
            task.id for task in rig.orchestrator.tasks if not task.is_done() and not task.error
        ] == [], f"{name}: a task was left with no explanation"


async def test_a_task_stranded_by_a_stop_is_skipped_rather_than_blocked(
    make_rig: RigBuilder,
) -> None:
    """The stop path owns undispatched work, stranded-or-not.

    ``a`` fails and asks for the stop on its way out, so ``b`` -- which could
    never run anyway -- and the untouched ``c`` are both left undispatched by
    the run. That decision is the run's, so it is reported as ``SKIPPED`` (with
    no attempt spent) rather than blamed on the plan and closed out BLOCKED.
    """
    rig_ref: list[Rig] = []

    async def fail_and_stop(task: TaskModel) -> None:
        # Ask for the stop, then fail. ``c`` is still queued behind the
        # concurrency cap, so it never starts; ``b`` was never ready at all.
        rig_ref[0].orchestrator.stop()
        raise RuntimeError("boom")

    rig = make_rig(
        plans=[
            make_plan(
                [
                    make_task("a", "write it"),
                    make_task("b", "after a", dependencies=["a"]),
                    make_task("c", "unrelated"),
                ],
                "fail, then stop",
            )
        ],
        verdicts=[passing()],
        worker_hook=fail_and_stop,
        settings_overrides={"max_parallel_tasks": 1, "max_cycles": 1},
    )
    rig_ref.append(rig)
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert [(task.id, task.status) for task in rig.orchestrator.tasks] == [
        ("a", TaskStatus.FAILED),
        ("b", TaskStatus.SKIPPED),
        ("c", TaskStatus.SKIPPED),
    ]
    assert [task.attempts for task in rig.orchestrator.tasks] == [1, 0, 0]
    assert rig.orchestrator.tasks[1].error == STOPPED_BEFORE_START


async def test_a_plan_where_everything_is_blocked_still_terminates(
    make_rig: RigBuilder,
) -> None:
    plan = make_plan(
        [
            make_task("a", "needs b", dependencies=["b"]),
            make_task("b", "needs a", dependencies=["a"]),
        ],
        "cycle",
    )
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await rig.orchestrator.run()
    assert only_payload(rig.events, "done") == {
        "reason": "Reached max_cycles (1)",
        "cycles": 1,
    }


def cycle_plus_dependent_plan() -> Plan:
    """``a`` and ``b`` deadlock each other and ``c`` needs ``a``.

    The shape that used to kill the process: the cycle walk abandoned ``a`` and
    ``b`` half-coloured, then the outer loop reached ``c``, saw a dependency in
    that half-finished state and raised ``ValueError`` out of ``run()``.
    """
    return make_plan(
        [
            make_task("a", "needs b", dependencies=["b"]),
            make_task("b", "needs a", dependencies=["a"]),
            make_task("c", "needs a", dependencies=["a"]),
        ],
        "cycle with a dependent",
    )


async def test_a_cycle_with_a_later_dependent_does_not_kill_the_run(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[cycle_plus_dependent_plan()],
        verdicts=[passing()],
        settings_overrides={"max_cycles": 1},
    )
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert only_payload(rig.events, "done") == {
        "reason": "Reached max_cycles (1)",
        "cycles": 1,
    }


async def test_a_cycle_with_a_later_dependent_blocks_every_task(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plans=[cycle_plus_dependent_plan()],
        verdicts=[passing()],
        settings_overrides={"max_cycles": 1},
    )
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert [task.status for task in rig.orchestrator.tasks] == [
        TaskStatus.BLOCKED,
        TaskStatus.BLOCKED,
        TaskStatus.BLOCKED,
    ]


async def test_a_task_depending_on_a_cycle_member_is_blocked_transitively(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[cycle_plus_dependent_plan()],
        verdicts=[passing()],
        settings_overrides={"max_cycles": 1},
    )
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert rig.orchestrator.tasks[2].error == "Dependency 'a' is blocked"


async def test_a_cycle_with_a_later_dependent_emits_the_documented_event_stream(
    make_rig: RigBuilder,
) -> None:
    """Nothing ran, but the run still finished and said so."""
    rig = make_rig(
        plans=[cycle_plus_dependent_plan()],
        verdicts=[passing()],
        settings_overrides={"max_cycles": 1},
    )
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert event_names(rig.events) == [
        "cycle_start",
        "plan",
        "test_result",
        "cycle_end",
        "done",
    ]
    assert rig.script.started == []


async def test_a_cycle_away_from_the_first_task_blocks_its_own_members_only(
    make_rig: RigBuilder,
) -> None:
    """A cycle that does not involve the first-listed task, plus a dependent."""
    plan = make_plan(
        [
            make_task("a", "unrelated"),
            make_task("b", "needs c", dependencies=["c"]),
            make_task("c", "needs b", dependencies=["b"]),
            make_task("d", "needs b", dependencies=["b"]),
        ],
        "cycle in the middle",
    )
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert [(task.id, task.status) for task in rig.orchestrator.tasks] == [
        ("a", TaskStatus.COMPLETED),
        ("b", TaskStatus.BLOCKED),
        ("c", TaskStatus.BLOCKED),
        ("d", TaskStatus.BLOCKED),
    ]


async def test_disjoint_cycles_and_their_dependents_are_all_blocked(
    make_rig: RigBuilder,
) -> None:
    plan = make_plan(
        [
            make_task("a1", "needs a2", dependencies=["a2"]),
            make_task("a2", "needs a1", dependencies=["a1"]),
            make_task("a3", "needs a1", dependencies=["a1"]),
            make_task("b1", "needs b2", dependencies=["b2"]),
            make_task("b2", "needs b1", dependencies=["b1"]),
            make_task("b3", "needs b1", dependencies=["b1"]),
        ],
        "two cycles",
    )
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert {task.status for task in rig.orchestrator.tasks} == {TaskStatus.BLOCKED}
    assert rig.script.started == []


async def test_a_long_dependency_chain_is_not_mistaken_for_a_cycle(
    make_rig: RigBuilder,
) -> None:
    """Deeper than the interpreter's recursion limit, so the walk must be iterative."""
    depth = 1200
    plan = make_plan(
        [
            make_task("t0", "root")
            if index == 0
            else make_task(f"t{index}", f"after t{index - 1}", dependencies=[f"t{index - 1}"])
            for index in range(depth)
        ],
        "deep chain",
    )
    rig = make_rig(
        plans=[plan],
        verdicts=[passing()],
        settings_overrides={"max_cycles": 1, "max_parallel_tasks": 4},
    )
    await asyncio.wait_for(rig.orchestrator.run(), timeout=30)
    assert len(rig.script.finished) == depth
    assert {task.status for task in rig.orchestrator.tasks} == {TaskStatus.COMPLETED}


async def test_an_unknown_dependency_next_to_a_cycle_does_not_kill_the_run(
    make_rig: RigBuilder,
) -> None:
    plan = make_plan(
        [
            make_task("a", "needs b", dependencies=["b"]),
            make_task("b", "needs a", dependencies=["a"]),
            make_task("ghosted", "needs a ghost", dependencies=["ghost"]),
            make_task("d", "needs ghosted", dependencies=["ghosted"]),
        ],
        "cycle and a dangling dep",
    )
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert {task.status for task in rig.orchestrator.tasks} == {TaskStatus.BLOCKED}
    assert only_payload(rig.events, "done")["cycles"] == 1


# --------------------------------------------------------------------------- #
# 5. Cycle bounds
# --------------------------------------------------------------------------- #


async def test_max_cycles_two_runs_exactly_two_cycles(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plan_for=lambda cycle: failing_plan(f"t{cycle}"),
        verdict_for=lambda cycle: failing(f"failure-{cycle}"),
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 2},
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 2


async def test_max_cycles_two_reports_the_cycle_bound_as_the_stop_reason(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plan_for=lambda cycle: failing_plan(f"t{cycle}"),
        verdict_for=lambda cycle: failing(f"failure-{cycle}"),
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 2},
    )
    await rig.orchestrator.run()
    assert only_payload(rig.events, "done")["reason"] == "Reached max_cycles (2)"


async def test_the_cycle_bound_is_checked_before_the_no_progress_guard(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plan_for=lambda cycle: failing_plan(),
        verdict_for=lambda cycle: failing("same failure every cycle"),
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 1, "max_consecutive_failures": 1},
    )
    await rig.orchestrator.run()
    assert only_payload(rig.events, "done")["reason"] == "Reached max_cycles (1)"


async def test_max_cycles_zero_does_not_self_limit(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plan_for=lambda cycle: failing_plan(f"t{cycle}"),
        verdict_for=lambda cycle: failing(f"failure-{cycle}"),
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 0, "max_consecutive_failures": 2},
    )
    rig.script.stop_on_test = 5
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 5


# --------------------------------------------------------------------------- #
# 6. No-progress guard
# --------------------------------------------------------------------------- #


async def test_identical_failing_cycles_stop_the_run(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plan_for=lambda cycle: failing_plan(),
        verdict_for=lambda cycle: failing("assert 1 == 2"),
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 0, "max_consecutive_failures": 2},
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 2


async def test_identical_failing_cycles_report_the_no_progress_stop_reason(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plan_for=lambda cycle: failing_plan(),
        verdict_for=lambda cycle: failing("assert 1 == 2"),
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 0, "max_consecutive_failures": 2},
    )
    await rig.orchestrator.run()
    assert only_payload(rig.events, "done")["reason"] == (
        "No progress after 2 identical failing cycles"
    )


async def test_a_different_failure_each_cycle_does_not_trip_the_guard(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plan_for=lambda cycle: failing_plan(),
        verdict_for=lambda cycle: failing(f"failure-{cycle}"),
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 0, "max_consecutive_failures": 2},
    )
    rig.script.stop_on_test = 4
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 4


async def test_a_different_failure_each_cycle_is_reported_as_a_manual_stop(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plan_for=lambda cycle: failing_plan(),
        verdict_for=lambda cycle: failing(f"failure-{cycle}"),
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 0, "max_consecutive_failures": 2},
    )
    rig.script.stop_on_test = 4
    await rig.orchestrator.run()
    assert only_payload(rig.events, "done")["reason"] == "Stop requested"


async def test_a_passing_cycle_clears_the_failure_history(make_rig: RigBuilder) -> None:
    blocked = make_plan([make_task("blocked", "stuck", dependencies=["ghost"])], "stuck")
    rig = make_rig(
        plans=[failing_plan(), blocked, failing_plan(), failing_plan()],
        verdict_for=lambda cycle: passing() if cycle == 2 else failing("boom"),
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 0, "max_consecutive_failures": 2},
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 4


async def test_the_guard_counts_only_the_failures_since_the_last_passing_cycle(
    make_rig: RigBuilder,
) -> None:
    blocked = make_plan([make_task("blocked", "stuck", dependencies=["ghost"])], "stuck")
    rig = make_rig(
        plans=[failing_plan(), blocked, failing_plan(), failing_plan()],
        verdict_for=lambda cycle: passing() if cycle == 2 else failing("boom"),
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 0, "max_consecutive_failures": 2},
    )
    await rig.orchestrator.run()
    assert only_payload(rig.events, "done") == {
        "reason": "No progress after 2 identical failing cycles",
        "cycles": 4,
    }


async def test_a_disabled_no_progress_guard_never_stops_the_loop(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plan_for=lambda cycle: failing_plan(),
        verdict_for=lambda cycle: failing("the same failure, every time"),
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 0, "max_consecutive_failures": 0},
    )
    rig.script.stop_on_test = 4
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 4


# --------------------------------------------------------------------------- #
# 7. Concurrency cap
# --------------------------------------------------------------------------- #


def five_independent_tasks() -> Plan:
    """A plan wide enough that ``max_parallel_tasks`` has to bite."""
    return make_plan([make_task(f"t{index}", f"do {index}") for index in range(5)], "wide")


async def test_peak_concurrency_never_exceeds_max_parallel_tasks(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[five_independent_tasks()],
        verdicts=[passing()],
        worker_hook=barrier_hook(2),
        settings_overrides={"max_parallel_tasks": 2},
    )
    await rig.orchestrator.run()
    assert rig.script.peak == 2


async def test_every_task_still_runs_under_the_concurrency_cap(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plans=[five_independent_tasks()],
        verdicts=[passing()],
        worker_hook=barrier_hook(2),
        settings_overrides={"max_parallel_tasks": 2},
    )
    await rig.orchestrator.run()
    assert sorted(rig.script.finished) == ["t0", "t1", "t2", "t3", "t4"]


async def test_max_parallel_tasks_of_one_serialises_every_task(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plans=[five_independent_tasks()],
        verdicts=[passing()],
        worker_hook=barrier_hook(1),
        settings_overrides={"max_parallel_tasks": 1},
    )
    await rig.orchestrator.run()
    assert rig.script.peak == 1


async def test_max_parallel_tasks_of_zero_is_treated_as_one(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plans=[five_independent_tasks()],
        verdicts=[passing()],
        worker_hook=barrier_hook(1),
        settings_overrides={"max_parallel_tasks": 0},
    )
    await rig.orchestrator.run()
    assert rig.script.peak == 1


# --------------------------------------------------------------------------- #
# 8. Retry budget (max_task_attempts escalation)
# --------------------------------------------------------------------------- #


async def test_a_tasks_first_failure_is_recorded_as_failed(make_rig: RigBuilder) -> None:
    rig = make_rig(worker_hook=always_fails, settings_overrides={"max_task_attempts": 2})
    task = make_task("t1", "do the thing")
    await dispatch(rig, task)
    assert task.status is TaskStatus.FAILED


async def test_a_task_is_not_skipped_on_its_first_failure(make_rig: RigBuilder) -> None:
    rig = make_rig(worker_hook=always_fails, settings_overrides={"max_task_attempts": 2})
    task = make_task("t1", "do the thing")
    await dispatch(rig, task)
    assert task.attempts == 1
    assert "Exceeded max_task_attempts" not in (task.error or "")


async def test_a_task_is_skipped_once_it_exceeds_max_task_attempts(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(worker_hook=always_fails, settings_overrides={"max_task_attempts": 2})
    task = make_task("t1", "do the thing")
    await dispatch(rig, task)
    task.status = TaskStatus.PENDING
    await dispatch(rig, task)
    assert task.status is TaskStatus.SKIPPED


async def test_a_skipped_task_explains_the_attempt_budget_it_exhausted(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(worker_hook=always_fails, settings_overrides={"max_task_attempts": 2})
    task = make_task("t1", "do the thing")
    await dispatch(rig, task)
    task.status = TaskStatus.PENDING
    await dispatch(rig, task)
    assert "Exceeded max_task_attempts (2)" in (task.error or "")


async def test_the_task_done_event_reports_the_skipped_status(make_rig: RigBuilder) -> None:
    rig = make_rig(worker_hook=always_fails, settings_overrides={"max_task_attempts": 2})
    task = make_task("t1", "do the thing")
    await dispatch(rig, task)
    task.status = TaskStatus.PENDING
    await dispatch(rig, task)
    assert payloads(rig.events, "task_done")[-1]["status"] is TaskStatus.SKIPPED


async def test_a_skipped_task_has_spent_every_attempt(make_rig: RigBuilder) -> None:
    rig = make_rig(worker_hook=always_fails, settings_overrides={"max_task_attempts": 2})
    task = make_task("t1", "do the thing")
    await dispatch(rig, task)
    task.status = TaskStatus.PENDING
    await dispatch(rig, task)
    assert task.attempts == 2


async def test_a_task_with_attempts_left_is_never_skipped(make_rig: RigBuilder) -> None:
    rig = make_rig(worker_hook=always_fails, settings_overrides={"max_task_attempts": 3})
    task = make_task("t1", "do the thing")
    await dispatch(rig, task)
    task.status = TaskStatus.PENDING
    await dispatch(rig, task)
    assert task.status is TaskStatus.FAILED


# --------------------------------------------------------------------------- #
# 9. Event contract
# --------------------------------------------------------------------------- #


async def test_a_happy_path_emits_the_documented_event_sequence(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(plans=[one_task_plan()], verdicts=[passing()])
    await rig.orchestrator.run()
    assert event_names(rig.events) == [
        "cycle_start",
        "plan",
        "task_start",
        "task_done",
        "test_result",
        "cycle_end",
        "done",
    ]


async def test_every_happy_path_event_carries_exactly_the_documented_keys(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(plans=[one_task_plan()], verdicts=[passing()])
    await rig.orchestrator.run()
    for name, payload in rig.events:
        assert set(payload) == EVENT_KEYS[name], f"{name} payload keys drifted"


async def test_the_plan_event_describes_every_planned_task(make_rig: RigBuilder) -> None:
    plan = make_plan(
        [
            make_task("a", "write it", files=["parser.py"]),
            make_task("b", "test it", dependencies=["a"]),
        ],
        "parser work",
    )
    rig = make_rig(plans=[plan], verdicts=[passing()])
    await rig.orchestrator.run()
    payload = only_payload(rig.events, "plan")
    assert payload["task_count"] == 2
    assert payload["tasks"] == [
        {
            "id": "a",
            "description": "write it",
            "files": ["parser.py"],
            "dependencies": [],
            "status": TaskStatus.PENDING,
        },
        {
            "id": "b",
            "description": "test it",
            "files": [],
            "dependencies": ["a"],
            "status": TaskStatus.PENDING,
        },
    ]


async def test_the_plan_event_reports_the_planners_rationale(make_rig: RigBuilder) -> None:
    rig = make_rig(plans=[one_task_plan(rationale="start small")], verdicts=[passing()])
    await rig.orchestrator.run()
    assert only_payload(rig.events, "plan")["rationale"] == "start small"


async def test_task_events_carry_the_id_and_the_task_description(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(plans=[make_plan([make_task("a", "write the parser")], "")], verdicts=[passing()])
    await rig.orchestrator.run()
    assert only_payload(rig.events, "task_start") == {
        "id": "a",
        "description": "write the parser",
    }


async def test_cycle_end_reports_the_completed_and_failed_ids(make_rig: RigBuilder) -> None:
    async def fail_only_the_broken_task(task: TaskModel) -> None:
        if task.id == "bad":
            raise RuntimeError("boom")

    rig = make_rig(
        plans=[make_plan([make_task("ok", "fine"), make_task("bad", "broken")], "")],
        verdicts=[failing("nope")],
        worker_hook=fail_only_the_broken_task,
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    assert only_payload(rig.events, "cycle_end") == {
        "cycle": 1,
        "passed": False,
        "completed": ["ok"],
        "failed": ["bad"],
    }


async def test_a_raising_on_event_callback_does_not_break_the_run(
    make_rig: RigBuilder,
) -> None:
    seen: list[str] = []

    def explode(name: str, payload: dict[str, Any]) -> None:
        seen.append(name)
        raise RuntimeError("listener is broken")

    rig = make_rig(
        plans=[one_task_plan("t1"), one_task_plan("t2")],
        verdicts=[failing("boom"), passing()],
        on_event=explode,
    )
    await rig.orchestrator.run()
    assert seen == event_names(rig.events)


async def test_a_raising_on_event_callback_still_lets_the_run_finish(
    make_rig: RigBuilder,
) -> None:
    def explode(name: str, payload: dict[str, Any]) -> None:
        raise RuntimeError("listener is broken")

    rig = make_rig(
        plans=[one_task_plan("t1"), one_task_plan("t2")],
        verdicts=[failing("boom"), passing()],
        on_event=explode,
    )
    await rig.orchestrator.run()
    assert event_names(rig.events)[-1] == "done"


# --------------------------------------------------------------------------- #
# 10. Resilience
# --------------------------------------------------------------------------- #


def planner_recovers(make_rig: RigBuilder) -> Rig:
    """A rig whose first cycle loses its plan to a planner failure, then recovers.

    The recovery plan is scripted because a cycle that planned nothing is no
    longer a finished run: the loop used to stop there and report the success
    message having done nothing, and now it carries on into a cycle 2 -- which
    therefore needs something to plan. Cycle 1 is the one under test here: the
    planner failure is still an ``error`` event, still falls back to an empty
    plan, and the cycle still runs the tester.
    """
    return make_rig(
        plans=[RuntimeError("planner exploded"), one_task_plan("t2")],
        verdicts=[passing(), passing()],
    )


def always_explodes(cycle: int) -> Plan:
    """``plan_for`` hook whose planner raises on every cycle, whatever the cycle."""
    raise RuntimeError("planner exploded")


def broken_planner_rig(make_rig: RigBuilder, **overrides: Any) -> Rig:
    """A rig whose planner fails on every cycle and whose tester passes every cycle.

    The nastiest shape there is: nothing runs, nothing is wrong with the
    workspace, and the verdict is green, so the only evidence that the run
    accomplished nothing is the planner's own ``error`` events.
    """
    return make_rig(
        plan_for=always_explodes,
        verdict_for=lambda cycle: passing(),
        settings_overrides={"max_cycles": 0, **overrides},
    )


async def test_a_planner_failure_is_reported_as_an_error_event(
    make_rig: RigBuilder,
) -> None:
    rig = planner_recovers(make_rig)
    await rig.orchestrator.run()
    assert only_payload(rig.events, "error") == {
        "message": "planner exploded",
        "where": "planner",
    }


async def test_a_planner_failure_falls_back_to_an_empty_plan(make_rig: RigBuilder) -> None:
    rig = planner_recovers(make_rig)
    await rig.orchestrator.run()
    assert rig.script.tester_calls[0].task_ids == ()


async def test_the_cycle_still_runs_the_tester_after_a_planner_failure(
    make_rig: RigBuilder,
) -> None:
    rig = planner_recovers(make_rig)
    await rig.orchestrator.run()
    # ``payloads(...)[0]`` rather than ``only_payload``: the run now carries on
    # into a second cycle (a cycle whose plan was lost is not a finished run),
    # so there are two verdicts. What matters is that the tester ran *in the
    # cycle whose planner failed* -- that is what cycle 1's verdict is.
    assert payloads(rig.events, "test_result")[0]["passed"] is True


async def test_a_planner_failure_cycle_runs_no_task_and_reports_no_success(
    make_rig: RigBuilder,
) -> None:
    """The Defect 1 repro: a cycle that planned nothing must not report success.

    Before the fix this run finished after a single cycle with the reason
    ``All tasks completed and tests passed``, having planned no task, run no
    task and verified nothing.
    """
    rig = broken_planner_rig(make_rig, max_consecutive_failures=1)
    with pytest.raises(PlannerUnavailableError):
        # ``wait_for`` is this file's standing liveness backstop: if the planner
        # budget stops bounding the run, the regression fails the test instead of
        # hanging the suite.
        await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert only_payload(rig.events, "done") == {
        "reason": (
            "Planner failed to produce a plan for 1 consecutive cycle "
            "(last error: planner exploded)"
        ),
        "cycles": 1,
    }
    assert rig.script.started == []
    assert "task_start" not in event_names(rig.events)


async def test_a_planner_failure_run_reports_its_reason_before_it_raises(
    make_rig: RigBuilder,
) -> None:
    """``done`` is emitted first, then the raise -- the event stream stays whole.

    The raise is what a caller that sees nothing but an exit status (the CLI
    passes no ``on_event``) needs in order not to call a do-nothing run a
    success, but it must not cost the terminal event its place in the stream.
    """
    rig = broken_planner_rig(make_rig, max_consecutive_failures=1)
    with pytest.raises(PlannerUnavailableError):
        # ``wait_for``: see the note on the Defect 1 repro above.
        await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert event_names(rig.events)[-1] == "done"
    assert only_payload(rig.events, "done")["reason"].startswith("Planner failed")


async def test_a_permanently_failing_planner_stops_under_an_unbounded_cycle_budget(
    make_rig: RigBuilder,
) -> None:
    """``max_cycles = 0`` must not turn a broken planner into an infinite run.

    The failure history cannot bound this one: the tester passes every cycle, so
    that history is cleared rather than appended to and the no-progress guard
    never has two identical signatures to compare.
    """
    rig = broken_planner_rig(make_rig, max_consecutive_failures=3)
    with pytest.raises(PlannerUnavailableError):
        await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert rig.orchestrator.cycles == 3
    assert only_payload(rig.events, "done") == {
        "reason": (
            "Planner failed to produce a plan for 3 consecutive cycles "
            "(last error: planner exploded)"
        ),
        "cycles": 3,
    }


async def test_the_whole_run_never_plans_a_task_when_the_planner_is_broken(
    make_rig: RigBuilder,
) -> None:
    rig = broken_planner_rig(make_rig, max_consecutive_failures=2)
    with pytest.raises(PlannerUnavailableError):
        await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert rig.orchestrator.tasks == []
    assert rig.script.planner_calls[0].goal == DEFAULT_GOAL


async def test_a_single_planner_failure_is_not_enough_to_stop_the_run(
    make_rig: RigBuilder,
) -> None:
    """One transient planner failure is retried, not treated as terminal."""
    rig = make_rig(
        plans=[RuntimeError("planner exploded"), one_task_plan("t2"), one_task_plan("t3")],
        verdict_for=lambda cycle: failing("boom"),
        settings_overrides={"max_cycles": 3, "max_consecutive_failures": 2},
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 2
    assert only_payload(rig.events, "done") == {
        "reason": "No progress after 2 identical failing cycles",
        "cycles": 2,
    }


async def test_a_disabled_no_progress_guard_keeps_a_failing_planner_retrying(
    make_rig: RigBuilder,
) -> None:
    """``max_consecutive_failures = 0`` disables the planner budget too.

    Same knob, same meaning: the user asked not to be stopped for making no
    progress, and a planner that keeps failing is the no-progress case the loop
    can still see nothing else about.
    """
    rig = broken_planner_rig(make_rig, max_consecutive_failures=0)
    rig.script.stop_on_plan = 3
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert rig.orchestrator.cycles == 3
    assert only_payload(rig.events, "done") == {"reason": "Stop requested", "cycles": 3}


async def test_a_tester_failure_is_reported_as_an_error_event(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[one_task_plan()],
        verdicts=[RuntimeError("tester exploded")],
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    assert only_payload(rig.events, "error") == {
        "message": "tester exploded",
        "where": "tester",
    }


async def test_a_tester_failure_counts_as_a_failed_test_run(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[one_task_plan()],
        verdicts=[RuntimeError("tester exploded")],
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    assert only_payload(rig.events, "test_result")["passed"] is False


async def test_a_worker_failure_is_recorded_as_a_failed_task(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[failing_plan()],
        verdicts=[passing()],
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    assert rig.orchestrator.tasks[0].status is TaskStatus.FAILED


async def test_the_task_done_event_reports_a_worker_failure(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[failing_plan()],
        verdicts=[passing()],
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    payload = only_payload(rig.events, "task_done")
    assert payload["status"] is TaskStatus.FAILED
    assert "boom" in (payload["error"] or "")


async def test_a_worker_failure_does_not_abort_the_cycle(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plans=[failing_plan()],
        verdicts=[passing()],
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    assert "cycle_end" in event_names(rig.events)


# --------------------------------------------------------------------------- #
# 11. Lifecycle
# --------------------------------------------------------------------------- #


async def test_is_running_is_false_before_the_run(make_rig: RigBuilder) -> None:
    rig = make_rig(plans=[one_task_plan()], verdicts=[passing()])
    assert rig.orchestrator.is_running is False


async def test_is_running_is_false_after_the_run(make_rig: RigBuilder) -> None:
    rig = make_rig(plans=[one_task_plan()], verdicts=[passing()])
    await rig.orchestrator.run()
    assert rig.orchestrator.is_running is False


async def test_is_running_is_true_while_a_task_is_executing(
    make_rig: RigBuilder,
) -> None:
    observed: list[bool] = []

    async def watch(task: TaskModel) -> None:
        observed.append(rig.script.is_running())

    rig = make_rig(plans=[one_task_plan()], verdicts=[passing()], worker_hook=watch)
    await rig.orchestrator.run()
    assert observed == [True]


async def test_stop_before_the_run_ends_it_without_a_cycle(make_rig: RigBuilder) -> None:
    rig = make_rig(plans=[one_task_plan()], verdicts=[passing()])
    rig.orchestrator.stop()
    await rig.orchestrator.run()
    assert only_payload(rig.events, "done") == {"reason": "Stop requested", "cycles": 0}


async def test_stop_before_the_run_never_calls_the_agents(make_rig: RigBuilder) -> None:
    rig = make_rig(plans=[one_task_plan()], verdicts=[passing()])
    rig.orchestrator.stop()
    await rig.orchestrator.run()
    assert rig.script.planner_calls == []


def self_stopping_rig(make_rig: RigBuilder) -> Rig:
    """A rig whose first cycle asks the orchestrator to stop, with no cycle bound."""
    rig = make_rig(
        plan_for=lambda cycle: failing_plan(f"t{cycle}"),
        verdict_for=lambda cycle: failing(f"failure-{cycle}"),
        settings_overrides={"max_cycles": 0},
    )
    rig.script.stop_on_test = 1
    return rig


async def test_stop_during_a_cycle_halts_at_the_next_cycle_boundary(
    make_rig: RigBuilder,
) -> None:
    rig = self_stopping_rig(make_rig)
    await rig.orchestrator.run()
    assert rig.orchestrator.cycles == 1
    assert only_payload(rig.events, "done") == {"reason": "Stop requested", "cycles": 1}


async def test_the_cycle_in_flight_still_finishes_after_a_stop(
    make_rig: RigBuilder,
) -> None:
    rig = self_stopping_rig(make_rig)
    await rig.orchestrator.run()
    assert "cycle_end" in event_names(rig.events)


def stop_mid_wave_rig(make_rig: RigBuilder) -> Rig:
    """A rig that asks the orchestrator to stop from inside its first worker.

    The plan is a single wave of independent tasks and the concurrency cap is 1,
    so only one worker is ever in flight and the rest of the wave is still
    waiting on the semaphore when the stop lands. Reachable from the CLI
    interrupt and from the web Stop button on any wave with more than one ready
    task; it used to spin in ``_execute_waves`` forever, leaving ``run()``
    suspended inside the gather with the CPU pegged.
    """
    rig_ref: list[Rig] = []
    stopped: list[str] = []

    async def stop_after_the_first_task(task: TaskModel) -> None:
        if not stopped:
            stopped.append(task.id)
            rig_ref[0].orchestrator.stop()

    rig = make_rig(
        plans=[five_independent_tasks()],
        verdicts=[passing()],
        worker_hook=stop_after_the_first_task,
        settings_overrides={"max_parallel_tasks": 1, "max_cycles": 1},
    )
    rig_ref.append(rig)
    return rig


async def test_stop_during_a_multi_task_wave_ends_the_run(make_rig: RigBuilder) -> None:
    rig = stop_mid_wave_rig(make_rig)
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert only_payload(rig.events, "done") == {"reason": "Stop requested", "cycles": 1}


async def test_stop_during_a_multi_task_wave_waits_for_the_task_in_flight(
    make_rig: RigBuilder,
) -> None:
    rig = stop_mid_wave_rig(make_rig)
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert rig.script.finished == ["t0"]


async def test_a_task_never_dispatched_because_of_a_stop_is_skipped(
    make_rig: RigBuilder,
) -> None:
    rig = stop_mid_wave_rig(make_rig)
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert [task.status for task in rig.orchestrator.tasks] == [
        TaskStatus.COMPLETED,
        TaskStatus.SKIPPED,
        TaskStatus.SKIPPED,
        TaskStatus.SKIPPED,
        TaskStatus.SKIPPED,
    ]


async def test_a_task_skipped_by_a_stop_explains_itself(make_rig: RigBuilder) -> None:
    rig = stop_mid_wave_rig(make_rig)
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert rig.orchestrator.tasks[1].error == "Run stopped before this task started"


async def test_a_task_skipped_by_a_stop_is_never_counted_as_an_attempt(
    make_rig: RigBuilder,
) -> None:
    rig = stop_mid_wave_rig(make_rig)
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert [task.attempts for task in rig.orchestrator.tasks] == [1, 0, 0, 0, 0]


async def test_a_stop_during_a_wave_reports_every_skipped_task_as_done(
    make_rig: RigBuilder,
) -> None:
    rig = stop_mid_wave_rig(make_rig)
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert [(payload["id"], payload["status"]) for payload in payloads(rig.events, "task_done")] == [
        ("t0", TaskStatus.COMPLETED),
        ("t1", TaskStatus.SKIPPED),
        ("t2", TaskStatus.SKIPPED),
        ("t3", TaskStatus.SKIPPED),
        ("t4", TaskStatus.SKIPPED),
    ]


async def test_a_stop_during_a_wave_leaves_no_task_pending(make_rig: RigBuilder) -> None:
    rig = stop_mid_wave_rig(make_rig)
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert all(task.is_terminal() for task in rig.orchestrator.tasks)


async def test_a_run_stopped_during_a_wave_keeps_its_state_file(make_rig: RigBuilder) -> None:
    rig = stop_mid_wave_rig(make_rig)
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert only_payload(rig.events, "done")["reason"] == "Stop requested"
    assert state_path(rig.settings).exists()


async def test_a_stop_ends_the_wave_loop_itself_not_the_progress_guard(
    monkeypatch: pytest.MonkeyPatch,
    make_rig: RigBuilder,
) -> None:
    """The stop check is what closes the loop, not the stalled-wave backstop.

    White-box on purpose: the forward-progress guard would also end this run,
    so without pinning *which* reason was used a missing stop check could hide
    behind it (leaving the skipped tasks with a "no progress" error and burning
    an extra, pointless wave pass).
    """
    reasons: list[str] = []
    original = Orchestrator._close_out_unstarted

    def spy(self: Orchestrator, reason: str) -> None:
        reasons.append(reason)
        original(self, reason)

    rig = stop_mid_wave_rig(make_rig)
    monkeypatch.setattr(Orchestrator, "_close_out_unstarted", spy)
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert reasons == ["Run stopped before this task started"]


async def test_every_wave_still_runs_when_no_stop_is_requested(make_rig: RigBuilder) -> None:
    """The stop check must not degenerate into "give up after one wave"."""
    plan = make_plan(
        [
            make_task("a", "root"),
            make_task("b", "after a", dependencies=["a"]),
            make_task("c", "after a", dependencies=["a"]),
            make_task("d", "after b", dependencies=["b"]),
        ],
        "three waves",
    )
    rig = make_rig(plans=[plan], verdicts=[passing()], settings_overrides={"max_cycles": 1})
    await rig.orchestrator.run()
    assert rig.script.start_snapshots == {
        "a": (),
        "b": ("a",),
        "c": ("a",),
        "d": ("a", "b", "c"),
    }


async def test_a_wave_that_makes_no_progress_is_closed_out_instead_of_spinning(
    monkeypatch: pytest.MonkeyPatch,
    make_rig: RigBuilder,
) -> None:
    """The forward-progress guard: a wave must change a status or end the loop.

    Driven with a worker that puts its task back exactly as it found it, which
    is the one way a wave can finish with the very same ready list it was handed.
    Without the guard ``_ready_tasks`` would keep returning that task and the
    loop would never end.
    """

    class IdleWorker:
        def __init__(
            self,
            task: TaskModel,
            client: LLMClient | None = None,
            settings: Settings | None = None,
        ) -> None:
            self.task = task

        async def run(self) -> str:
            self.task.status = TaskStatus.PENDING
            return ""

    rig = make_rig(settings_overrides={"max_cycles": 1})
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", IdleWorker)
    task = make_task("t1", "undo its own work")
    rig.orchestrator.tasks = [task]
    await asyncio.wait_for(rig.orchestrator._execute_waves(), timeout=5)
    assert task.status is TaskStatus.SKIPPED
    assert "no progress" in (task.error or "")


async def test_a_stalled_wave_still_lets_the_run_finish(
    monkeypatch: pytest.MonkeyPatch,
    make_rig: RigBuilder,
) -> None:
    class IdleWorker:
        def __init__(
            self,
            task: TaskModel,
            client: LLMClient | None = None,
            settings: Settings | None = None,
        ) -> None:
            self.task = task

        async def run(self) -> str:
            self.task.status = TaskStatus.PENDING
            return ""

    rig = make_rig(
        plan_for=lambda cycle: one_task_plan(f"t{cycle}"),
        verdicts=[passing()],
        settings_overrides={"max_cycles": 1},
    )
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", IdleWorker)
    await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert rig.orchestrator.tasks[0].status is TaskStatus.SKIPPED
    assert only_payload(rig.events, "done") == {
        "reason": "Reached max_cycles (1)",
        "cycles": 1,
    }


def blocking_rig(make_rig: RigBuilder) -> Rig:
    """A rig whose single task never finishes, so the run can be cancelled."""
    rig = make_rig(plans=[one_task_plan()], verdicts=[passing()])
    rig.script.block_tasks = asyncio.Event()
    return rig


async def test_cancellation_propagates_out_of_run(make_rig: RigBuilder) -> None:
    rig = blocking_rig(make_rig)
    with pytest.raises(asyncio.CancelledError):
        await run_until_cancelled(rig)


async def test_cancellation_emits_done_before_propagating(make_rig: RigBuilder) -> None:
    rig = blocking_rig(make_rig)
    with pytest.raises(asyncio.CancelledError):
        await run_until_cancelled(rig)
    assert only_payload(rig.events, "done") == {"reason": "Cancelled", "cycles": 1}


async def test_cancellation_clears_the_running_flag(make_rig: RigBuilder) -> None:
    rig = blocking_rig(make_rig)
    with pytest.raises(asyncio.CancelledError):
        await run_until_cancelled(rig)
    assert rig.orchestrator.is_running is False


#: A self-contained program that must stop itself, or fail to.
#:
#: ``call_later`` is the cheapest possible same-loop interruption: it fires only
#: if the orchestrator's cycle loop hands control back to the event loop. It runs
#: in its own process precisely because that is the only honest way to test it --
#: a loop that never yields starves every coroutine on its own loop, so nothing
#: inside the test could ever notice, not even ``asyncio.wait_for``. The GIL does
#: not help: from another *thread* the flag would be seen anyway, which would
#: make a non-cooperative loop look cooperative.
SPINNING_PROGRAM = """\
import asyncio
import sys
import tempfile
from pathlib import Path

from furrow.config import Plan, Settings, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class Unusable(LLMClient):
    async def complete(self, *args: object, **kwargs: object) -> str:
        raise RuntimeError("planner exploded")


async def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            workspace=Path(tmp),
            state_file=Path(tmp) / ".furrow" / "state.json",
            # Both progress guards off, so this run is meant to spin forever.
            # Nothing else may end it: not the cycle bound, not the failure
            # history, not an empty plan. Only the event loop getting control
            # back at a cycle boundary can.
            max_cycles=0,
            max_consecutive_failures=0,
        )
        orchestrator = Orchestrator(goal="spin", client=Unusable(settings), settings=settings)

        async def plan(goal: str, context: str = "") -> Plan:
            raise RuntimeError("planner exploded")

        async def test(goal: str, tasks: list) -> TestResult:
            return TestResult(passed=True, summary="ok")

        orchestrator.planner.plan = plan
        orchestrator.tester.run = test
        asyncio.get_running_loop().call_later(0.5, orchestrator.stop)
        await orchestrator.run()
        print(f"STOPPED after {orchestrator.cycles} cycles")
        return 0


sys.exit(asyncio.run(main()))
"""


def test_a_run_whose_cycles_never_suspend_still_yields_to_the_event_loop(
    tmp_path: Path,
) -> None:
    """The loop hands control back at every cycle boundary, even when none waits.

    A cycle whose planner, tasks and tester all finish immediately contains no
    real suspension point, so without a yield at the boundary the loop spins
    without ever suspending -- and a coroutine that never yields can be neither
    timed out, cancelled nor stopped, which leaves ``stop()``, ``Ctrl+C`` and
    ``asyncio.wait_for`` with nothing to interrupt. It also makes an unbounded
    broken-planner run untestable: the regression could only ever be caught by a
    CI timeout, which is how it got here in the first place.
    """
    program = tmp_path / "spinning_loop.py"
    program.write_text(SPINNING_PROGRAM)
    # The child needs the package importable exactly as this process has it:
    # the repo root (for ``furrow``) plus whatever PYTHONPATH already carries
    # (``pydantic`` and friends). No network, no API key, no real LLM.
    repo_root = str(Path(orchestrator_module.__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [repo_root, env.get("PYTHONPATH", "")]))
    result = subprocess.run(
        [sys.executable, str(program)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )
    assert result.returncode == 0, (
        f"the run could not stop itself: {result.stdout[-2000:]}{result.stderr[-2000:]}"
    )
    assert "STOPPED after" in result.stdout, result.stdout


# --------------------------------------------------------------------------- #
# 12. Persistence
# --------------------------------------------------------------------------- #


def interrupted_rig(make_rig: RigBuilder) -> Rig:
    """A rig whose single cycle fails its task and its verdict, then hits the bound.

    A run that stops on ``max_cycles`` (rather than completing) is exactly the
    case that must leave the persisted state behind.
    """
    return make_rig(
        plans=[failing_plan()],
        verdicts=[failing("boom")],
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 1},
    )


async def test_a_state_file_is_written_after_a_cycle(make_rig: RigBuilder) -> None:
    rig = interrupted_rig(make_rig)
    await rig.orchestrator.run()
    assert state_path(rig.settings).exists()


async def test_the_state_file_is_readable_by_a_state_store(make_rig: RigBuilder) -> None:
    rig = interrupted_rig(make_rig)
    await rig.orchestrator.run()
    assert StateStore(rig.settings).load() is not None


async def test_the_state_file_records_the_goal_and_the_cycle_count(
    make_rig: RigBuilder,
) -> None:
    rig = interrupted_rig(make_rig)
    await rig.orchestrator.run()
    loaded = StateStore(rig.settings).load()
    assert loaded is not None
    assert (loaded.goal, loaded.cycles) == (DEFAULT_GOAL, 1)


async def test_the_state_file_records_every_cycle_of_the_run(make_rig: RigBuilder) -> None:
    # This used to assert ``== [2]``: each cycle REPLACED the history instead of
    # appending to it, so a two-cycle run persisted exactly one record. The old
    # expectation encoded that data loss, and the "latest cycle" coverage it
    # gave is kept below.
    rig = make_rig(
        plans=[failing_plan(), failing_plan()],
        verdicts=[failing("boom"), failing("bang")],
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 2},
    )
    await rig.orchestrator.run()
    loaded = StateStore(rig.settings).load()
    assert loaded is not None
    assert [record.cycle for record in loaded.history] == [1, 2]
    assert loaded.history[-1].failures == ["bang"]


async def test_the_state_file_history_keeps_the_first_cycle_too(make_rig: RigBuilder) -> None:
    rig = make_rig(
        plans=[failing_plan(), failing_plan(), failing_plan()],
        verdicts=[failing("boom"), failing("bang"), failing("bump")],
        worker_hook=always_fails,
        settings_overrides={"max_cycles": 3},
    )
    await rig.orchestrator.run()
    loaded = StateStore(rig.settings).load()
    assert loaded is not None
    assert [record.failures for record in loaded.history] == [
        ["boom"],
        ["bang"],
        ["bump"],
    ]


async def test_the_state_file_history_is_capped_and_drops_the_oldest_cycles(
    make_rig: RigBuilder,
) -> None:
    cycles = orchestrator_module.MAX_HISTORY_RECORDS + 10
    rig = make_rig(
        plan_for=lambda cycle: failing_plan(f"t{cycle}"),
        verdict_for=lambda cycle: failing(f"failure-{cycle}"),
        worker_hook=always_fails,
        settings_overrides={"max_cycles": cycles},
    )
    await rig.orchestrator.run()
    loaded = StateStore(rig.settings).load()
    assert loaded is not None
    assert loaded.cycles == cycles
    assert [record.cycle for record in loaded.history] == list(
        range(cycles - orchestrator_module.MAX_HISTORY_RECORDS + 1, cycles + 1)
    )


async def test_the_state_file_records_the_completed_tasks_of_the_cycle(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[one_task_plan()],
        verdicts=[failing("boom")],
        settings_overrides={"max_cycles": 1},
    )
    await rig.orchestrator.run()
    loaded = StateStore(rig.settings).load()
    assert loaded is not None
    assert loaded.history[0].completed == ["t1"]


async def test_the_state_file_is_removed_after_a_successful_run(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(plans=[one_task_plan()], verdicts=[passing()])
    await rig.orchestrator.run()
    assert not state_path(rig.settings).exists()


async def test_the_state_file_survives_a_run_stopped_by_max_cycles(
    make_rig: RigBuilder,
) -> None:
    rig = interrupted_rig(make_rig)
    await rig.orchestrator.run()
    assert only_payload(rig.events, "done")["reason"] == "Reached max_cycles (1)"
    assert state_path(rig.settings).exists()


async def test_the_state_file_is_written_inside_the_temporary_workspace(
    make_rig: RigBuilder,
    tmp_path: Path,
) -> None:
    rig = interrupted_rig(make_rig)
    await rig.orchestrator.run()
    assert state_path(rig.settings).is_relative_to(tmp_path)


async def test_a_run_that_never_got_a_plan_keeps_its_state_file(make_rig: RigBuilder) -> None:
    """``clear()`` is for genuine success only: this run achieved nothing."""
    rig = broken_planner_rig(make_rig, max_consecutive_failures=1)
    with pytest.raises(PlannerUnavailableError):
        # ``wait_for``: see the note on the Defect 1 repro above.
        await asyncio.wait_for(rig.orchestrator.run(), timeout=5)
    assert state_path(rig.settings).exists()
    loaded = StateStore(rig.settings).load()
    assert loaded is not None
    assert loaded.tasks == []


async def test_a_completing_cycle_clears_the_state_written_by_the_failed_one(
    make_rig: RigBuilder,
) -> None:
    rig = make_rig(
        plans=[failing_plan(), one_task_plan()],
        verdicts=[failing("boom"), passing()],
        settings_overrides={"max_cycles": 2},
    )
    await rig.orchestrator.run()
    assert not state_path(rig.settings).exists()