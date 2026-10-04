"""Loop-control regression suite for :mod:`furrow.core.orchestrator`.

The bug this file exists to kill
--------------------------------
``Orchestrator._get_tasks()`` shipped as a stub returning ``[]``. Every
``_is_done()`` call therefore saw an empty plan, "is every task finished?" was
vacuously true, and the loop reported ``goal complete`` on every single cycle:
Furrow's flagship infinite development loop silently halted after exactly one
cycle *and* announced success even when every task and every test had failed.

The contract pinned here
------------------------
``Orchestrator.run()`` returns ``None`` and exactly one termination condition
must fire, recorded in ``stopped_reason``:

* ``goal complete`` -- every planned task COMPLETED **and** the suite passed with
  ``verified=True``.
* ``max_cycles`` -- ``settings.max_cycles`` cycles ran (0 means unlimited).
* ``stalled`` -- ``max_stalled_cycles`` consecutive cycles changed no file and
  returned an identical ``(passed, failures)`` verdict. The halt fires one cycle
  *after* the threshold because the first no-change cycle has no previous verdict
  to compare against; the message reports the *real* streak length (tracked by
  ``_no_progress``), so it always matches the cycles that actually ran.
* unverifiable -- ``verified=False`` halts with an "unverifiable, not passing"
  diagnostic; never claims completion, never starts a repair cycle.
* ``planner failed`` -- ``MAX_CONSECUTIVE_PLANNER_FAILURES`` unusable plans in a row.
* ``planner returned no tasks`` -- nothing left to decompose.
* unschedulable plan (dependency cycle) -- reported, never deadlocked.
* ``cancelled`` -- recorded, then ``CancelledError`` re-raised.

Execution invariants pinned alongside them: dependency order, no worker built for
a task whose dependency failed, ``max_parallel_tasks`` as a hard concurrency cap,
a worker that changed nothing is FAILED, a raising worker fails only its own task,
and ``goal`` is never mutated by a repair cycle.

Stall arithmetic (deliberate, asserted as-is): ``_last_verdict`` starts as
``None``, so the *first* no-change cycle resets ``_stalled`` to 0 instead of
incrementing it. ``max_stalled_cycles=3`` therefore fires on the 4th cycle.
The halt message now derives from ``_no_progress`` (the honest streak of cycles
that changed no file and repeated the verdict, including the verdict-less first
cycle), so it correctly reports 4 instead of the misleading 3.

Technique
---------
``Orchestrator`` constructs its three agents itself, so every test monkeypatches
``PlannerAgent`` / ``WorkerAgent`` / ``TesterAgent`` *on the orchestrator module*
with scripted fakes driven by a :class:`World`. No network, no subprocesses, no
API keys and no writes outside pytest's ``tmp_path``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult
from furrow.core.orchestrator import (
    COMPLETE_PREFIX,
    DEFAULT_MAX_STALLED_CYCLES,
    MAX_CONSECUTIVE_PLANNER_FAILURES,
    Orchestrator,
)
from furrow.llm import LLMClient
from tests.conftest import FakeLLM, make_settings_for

#: The user's objective. Multiline on purpose: a repair cycle must never edit it.
GOAL = "Support TOML config files\nwithout regressing YAML loading."

#: Cycles a default-configured stall takes: see the stall arithmetic note above.
STALL_CYCLES = 4  # = DEFAULT_MAX_STALLED_CYCLES + 1 (off-by-one from None verdict)

#: A generous ceiling that still turns a runaway loop into a fast failure.
NO_HANG_TIMEOUT = 5.0

COMPLETE_REASON = f"{COMPLETE_PREFIX}: every planned task applied its edits and tests pass"

#: Distinguishes "the ``failures`` keyword was never passed" from "it was passed
#: empty". :class:`FakeWorker` records ``None`` for the former, ``()`` for the
#: latter -- both are "no repair context", but only one is the historical call.
_OMITTED = object()


# --------------------------------------------------------------------------- #
# Scripted world + fake agents
# --------------------------------------------------------------------------- #


@dataclass
class WorkerScript:
    """Scripted outcome of one fake worker run.

    ``files=None`` means "edit one file derived from the task id" (the happy
    path); ``files=()`` means "apply nothing", which the orchestrator must treat
    as FAILED. ``boom`` raises out of ``run()`` instead of returning.
    """

    files: tuple[str, ...] | None = None
    error: str | None = None
    boom: Exception | None = None
    summary: str = "implemented the task"


@dataclass
class World:
    """Script state shared by the fake planner, workers and tester.

    ``plans`` / ``tester_results`` are indexed by call number and the **last entry
    repeats** for any further cycle, so a test only scripts the interesting
    prefix. Every script must be non-empty: an empty one raises loudly instead of
    silently turning into a planner or tester failure.
    """

    plans: list[Plan | Exception] = field(default_factory=list)
    tester_results: list[TestResult] = field(default_factory=list)
    worker_scripts: dict[str, WorkerScript] = field(default_factory=dict)
    #: Recorded as ``start:<id>`` / ``done:<id>`` / ``raise:<id>``.
    order: list[str] = field(default_factory=list)
    #: Task ids whose worker was *constructed*; proves skipped tasks never built one.
    constructed: list[str] = field(default_factory=list)
    #: ``(task_id, failures)`` per worker construction, in construction order. The
    #: second element is ``None`` when the ``failures`` keyword was omitted entirely,
    #: so "not a repair cycle" is distinguishable from "an explicitly empty context".
    worker_constructions: list[tuple[str, tuple[str, ...] | None]] = field(default_factory=list)
    #: The focus text handed to the planner, one entry per cycle.
    planner_focuses: list[str] = field(default_factory=list)
    #: The goal text handed to the tester, one entry per cycle.
    tester_goals: list[str] = field(default_factory=list)
    planner_calls: int = 0
    tester_calls: int = 0
    #: Set by the fake tester on entry, so a test can await it before cancelling.
    tester_entered: asyncio.Event = field(default_factory=asyncio.Event)
    #: When set, the fake tester parks on it until the test opens it.
    tester_gate: asyncio.Event | None = None
    in_flight: int = 0
    peak_in_flight: int = 0


def _scripted(items: list[Any], call: int, what: str) -> Any:
    """The entry for 1-based call ``call``; the final entry repeats forever."""
    if not items:
        raise AssertionError(f"World.{what} is empty: script at least one entry")
    return items[min(call, len(items)) - 1]


def install_fakes(monkeypatch: pytest.MonkeyPatch, world: World) -> None:
    """Replace the three agents the orchestrator builds with scripted fakes.

    The fakes implement only what :class:`Orchestrator` actually touches:
    ``PlannerAgent.plan(focus)``, ``WorkerAgent(task=..., client=...)`` with
    ``run()``/``changed_files``/``error``, and ``TesterAgent(client=...).run(...)``.
    """

    class FakePlanner:
        def __init__(
            self, client: LLMClient | None = None, settings: Settings | None = None
        ) -> None:
            self.client = client

        async def plan(self, focus: str) -> Plan:
            world.planner_calls += 1
            world.planner_focuses.append(focus)
            scripted = _scripted(world.plans, world.planner_calls, "plans")
            if isinstance(scripted, Exception):
                raise scripted
            return scripted

    class FakeWorker:
        def __init__(
            self,
            task: TaskModel,
            client: LLMClient | None = None,
            settings: Settings | None = None,
            *,
            failures: Any = _OMITTED,
        ) -> None:
            self.task = task
            self.changed_files: list[str] = []
            self.deleted_files: list[str] = []
            self.error: str | None = None
            world.constructed.append(task.id)
            world.worker_constructions.append(
                (task.id, None if failures is _OMITTED else tuple(failures or ()))
            )

        async def run(self) -> str:
            script = world.worker_scripts.get(self.task.id, WorkerScript())
            files = script.files if script.files is not None else (f"src/{self.task.id}.py",)
            world.order.append(f"start:{self.task.id}")
            world.in_flight += 1
            world.peak_in_flight = max(world.peak_in_flight, world.in_flight)
            try:
                await asyncio.sleep(0)  # one yield so genuine overlap is observable
                if script.boom is not None:
                    world.order.append(f"raise:{self.task.id}")
                    raise script.boom
                self.changed_files = list(files)
                self.error = script.error
                world.order.append(f"done:{self.task.id}")
                return script.summary
            finally:
                world.in_flight -= 1

    class FakeTester:
        def __init__(
            self, client: LLMClient | None = None, settings: Settings | None = None
        ) -> None:
            self.client = client

        async def run(self, goal: str, tasks: list[TaskModel], **kwargs: Any) -> TestResult:
            del tasks, kwargs  # the fake judges from the script, not from the plan
            world.tester_calls += 1
            world.tester_goals.append(goal)
            world.tester_entered.set()
            if world.tester_gate is not None:
                await world.tester_gate.wait()
            result = _scripted(world.tester_results, world.tester_calls, "tester_results")
            assert isinstance(result, TestResult)
            return result

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", FakePlanner)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", FakeTester)


# --------------------------------------------------------------------------- #
# Small constructors
# --------------------------------------------------------------------------- #


def task(
    task_id: str,
    *,
    files: list[str] | None = None,
    dependencies: list[str] | None = None,
) -> TaskModel:
    """One planned task."""
    return TaskModel(
        id=task_id,
        description=f"implement {task_id}",
        files=files or [],
        dependencies=dependencies or [],
    )


def plan(*tasks: TaskModel, rationale: str = "scripted plan") -> Plan:
    """A plan for the given tasks."""
    return Plan(tasks=list(tasks), rationale=rationale)


def passing(summary: str = "12 passed") -> TestResult:
    """A green, verified verdict."""
    return TestResult(passed=True, summary=summary, failures=[], verified=True)


def failing(*failures: str, summary: str = "1 failed") -> TestResult:
    """A red, verified verdict -- real signal, so the loop must try a repair."""
    return TestResult(passed=False, summary=summary, failures=list(failures), verified=True)


def unverified(summary: str = "No test runner detected") -> TestResult:
    """No runner was found: ``passed=True`` carries no signal and must not count.

    ``passed=True`` on purpose -- it is the strongest version of the trap: the
    loop must trust ``verified`` over ``passed``.
    """
    return TestResult(passed=True, summary=summary, failures=[], verified=False)


def build(
    goal: str,
    world: World,
    *,
    settings: Settings | None = None,
    **kwargs: Any,
) -> Orchestrator:
    """An orchestrator on ``world``'s fakes, driven by a network-free FakeLLM.

    ``kwargs`` goes to ``Orchestrator`` (``on_event``, ``max_stalled_cycles``).
    The ``FakeLLM`` is never prompted -- both it and the real ``LLMClient`` would
    only supply ``client.settings`` -- but it keeps ``Orchestrator.__init__`` from
    ever reaching for a real provider client.
    """
    client = FakeLLM(default="", settings=settings or make_settings_for())
    return Orchestrator(goal, client=client, **kwargs)


def record(events: list[tuple[str, dict[str, Any]]]) -> Callable[[str, dict[str, Any]], None]:
    """An ``on_event`` hook appending ``(name, payload)`` to ``events``."""

    def _record(name: str, payload: dict[str, Any]) -> None:
        events.append((name, payload))

    return _record


def by_id(tasks: list[TaskModel]) -> dict[str, TaskModel]:
    """Index tasks by id for status assertions."""
    return {item.id: item for item in tasks}


# --------------------------------------------------------------------------- #
# 1-2. Completion requires BOTH every task applied and a green verified suite
# --------------------------------------------------------------------------- #


async def test_completes_when_all_tasks_done_and_tests_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(plans=[plan(task("a"), task("b"))], tester_results=[passing()])
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_parallel_tasks=2))

    assert await orch.run() is None

    assert orch.stopped_reason == COMPLETE_REASON
    assert orch.stopped_reason.startswith(COMPLETE_PREFIX)
    assert [item.status for item in orch.tasks] == [TaskStatus.COMPLETED, TaskStatus.COMPLETED]
    assert sorted(orch.changed_files) == ["src/a.py", "src/b.py"]
    assert len(orch.test_results) == 1 and orch.test_results[0].passed is True
    # It stopped on the completion verdict instead of looping again.
    assert (orch.cycles, world.planner_calls, world.tester_calls) == (1, 1, 1)


async def test_incomplete_goal_is_not_reported_as_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original bug: a cycle where everything fails must never say "complete"."""
    events: list[tuple[str, dict[str, Any]]] = []
    world = World(
        plans=[plan(task("a"))],
        worker_scripts={"a": WorkerScript(files=())},
        tester_results=[failing("tests/test_a.py::test_a FAILED")],
    )
    install_fakes(monkeypatch, world)
    orch = build(
        GOAL,
        world,
        settings=make_settings_for(max_cycles=1),
        on_event=record(events),
    )

    await orch.run()

    reason = orch.stopped_reason
    assert reason is not None
    assert reason == "reached the max_cycles limit of 1 cycle(s)"
    assert not reason.startswith(COMPLETE_PREFIX)
    assert orch.tasks[0].status is TaskStatus.FAILED
    assert orch.test_results[0].passed is False
    halt = [payload for name, payload in events if name == "halt"]
    assert len(halt) == 1
    assert halt[0]["complete"] is False
    assert orch.changed_files == []


async def test_passing_suite_with_unapplied_task_is_not_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A green suite is not enough: every planned task must also have landed."""
    world = World(
        plans=[plan(task("ok"), task("noop"))],
        worker_scripts={"noop": WorkerScript(files=())},
        tester_results=[passing()],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=2))

    await orch.run()

    assert [item.status for item in orch.tasks] == [TaskStatus.COMPLETED, TaskStatus.FAILED]
    reason = orch.stopped_reason
    assert reason == "reached the max_cycles limit of 2 cycle(s)"
    assert not reason.startswith(COMPLETE_PREFIX)
    # The second cycle was handed the untouched goal plus the unfinished task id.
    assert world.planner_focuses[1].startswith(GOAL)
    assert "must still land: noop" in world.planner_focuses[1]


# --------------------------------------------------------------------------- #
# 3-5. Repair cycles, goal immutability, max_cycles
# --------------------------------------------------------------------------- #


async def test_retry_cycles_preserve_the_original_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = "tests/test_toml.py::test_nested_table FAILED"
    world = World(
        plans=[plan(task("a")) for _ in range(3)],
        tester_results=[failing(failure), failing(failure), passing()],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=0))

    await orch.run()

    assert orch.cycles == 3
    assert orch.goal == GOAL  # byte-identical: repair cycles never edit the goal
    assert world.tester_goals == [GOAL, GOAL, GOAL]
    # The repair focus carries the goal verbatim plus the current failure text.
    assert orch.focus.startswith(GOAL)
    assert failure in orch.focus
    assert world.planner_focuses[1] == world.planner_focuses[2]
    assert failure in world.planner_focuses[2]
    assert orch.stopped_reason == COMPLETE_REASON


async def test_max_cycles_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(
        plans=[plan(task("a"))],
        tester_results=[failing("tests/test_a.py::test_a FAILED")],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=2))

    await orch.run()

    assert orch.cycles == 2
    assert orch.stopped_reason == "reached the max_cycles limit of 2 cycle(s)"
    assert not orch.stopped_reason.startswith(COMPLETE_PREFIX)
    assert world.planner_calls == 2 and world.tester_calls == 2
    assert len(orch.test_results) == 2


async def test_max_cycles_zero_means_unlimited(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = "tests/test_b.py::test_b FAILED"
    world = World(
        plans=[plan(task("a")) for _ in range(3)],
        tester_results=[failing(failure), failing(failure), passing()],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=0))

    await asyncio.wait_for(orch.run(), NO_HANG_TIMEOUT)

    # Not cut short at 1 or 2 cycles (the finite-limit cases), and still finite.
    assert (orch.cycles, world.tester_calls) == (3, 3)
    assert orch.stopped_reason == COMPLETE_REASON
    assert [result.passed for result in orch.test_results] == [False, False, True]


# --------------------------------------------------------------------------- #
# 6-8. Dependencies
# --------------------------------------------------------------------------- #


async def test_dependency_ordering_waits_for_the_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(
        plans=[plan(task("a"), task("b", dependencies=["a"]))],
        tester_results=[passing()],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world)

    await orch.run()

    assert world.order == ["start:a", "done:a", "start:b", "done:b"]
    assert world.order.index("done:a") < world.order.index("start:b")
    assert orch.stopped_reason == COMPLETE_REASON
    assert orch.changed_files == ["src/a.py", "src/b.py"]


async def test_task_with_failed_dependency_is_skipped_and_marked_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(
        plans=[plan(task("a"), task("b", dependencies=["a"]))],
        worker_scripts={"a": WorkerScript(boom=RuntimeError("model exploded"))},
        tester_results=[failing("tests/test_a.py::test_a FAILED")],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=1))

    await orch.run()

    assert world.constructed == ["a"]  # b's worker was never even built
    assert world.order == ["start:a", "raise:a"]  # and never ran
    tasks = by_id(orch.tasks)
    assert tasks["a"].status is TaskStatus.FAILED
    assert tasks["b"].status is TaskStatus.FAILED
    assert tasks["b"].result is not None
    assert "dependency a did not complete" in tasks["b"].result
    assert "src/b.py" not in orch.changed_files


async def test_dependency_cycle_fails_the_plan_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(
        plans=[plan(task("a", dependencies=["b"]), task("b", dependencies=["a"]))],
        tester_results=[passing()],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=0))

    await asyncio.wait_for(orch.run(), NO_HANG_TIMEOUT)  # must not deadlock

    reason = orch.stopped_reason
    assert reason is not None
    assert "the plan could not be scheduled" in reason
    assert "dependency cycle" in reason
    assert not reason.startswith(COMPLETE_PREFIX)
    assert orch.cycles == 1
    assert world.constructed == []  # nothing was scheduled at all
    assert world.tester_calls == 0  # the cycle never reached the suite
    assert orch.test_results == []


# --------------------------------------------------------------------------- #
# 9-10. No-progress and unverifiable halts
# --------------------------------------------------------------------------- #


async def test_stall_detection_halts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The halt lands on cycle 4; the message now correctly reports 4 (was 3)."""
    world = World(
        plans=[plan(task("a")) for _ in range(STALL_CYCLES)],
        worker_scripts={"a": WorkerScript(files=())},
        tester_results=[failing("tests/test_a.py::test_a FAILED")],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world)

    await asyncio.wait_for(orch.run(), NO_HANG_TIMEOUT)

    reason = orch.stopped_reason
    assert reason is not None
    # The message now reports the _no_progress streak (real cycles that ran), not
    # the _stalled counter which is off-by-one due to the initial None verdict.
    assert reason == (
        f"stalled: {STALL_CYCLES} consecutive cycles changed no file "
        "and produced identical test failures"
    )
    assert not reason.startswith(COMPLETE_PREFIX)
    # Halt timing is unchanged: it fires on cycle 4 (DEFAULT_MAX_STALLED_CYCLES + 1).
    assert orch.cycles == STALL_CYCLES == DEFAULT_MAX_STALLED_CYCLES + 1
    assert world.planner_calls == STALL_CYCLES
    assert world.tester_calls == STALL_CYCLES  # a further cycle did not run
    assert orch.changed_files == []
    assert orch.tasks[0].status is TaskStatus.FAILED


async def test_unverified_tests_halt_without_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(plans=[plan(task("a"))], tester_results=[unverified()])
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=0))

    await asyncio.wait_for(orch.run(), NO_HANG_TIMEOUT)

    reason = orch.stopped_reason
    assert reason is not None
    assert "no detectable test runner" in reason
    assert "unverifiable, not passing" in reason
    assert not reason.startswith(COMPLETE_PREFIX)
    assert orch.cycles == 1
    assert world.tester_calls == 1  # no repair cycle was started
    assert world.planner_calls == 1
    assert orch.test_results[0].verified is False
    assert world.planner_focuses == [GOAL]  # focus left untouched


# --------------------------------------------------------------------------- #
# 11. Planner failure
# --------------------------------------------------------------------------- #


async def test_planner_exception_does_not_escape_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    world = World(plans=[RuntimeError("planner exploded")])
    install_fakes(monkeypatch, world)
    orch = build(
        GOAL,
        world,
        settings=make_settings_for(max_cycles=0),
        on_event=record(events),
    )

    assert await orch.run() is None  # no raw traceback reaches the caller

    reason = orch.stopped_reason
    assert reason is not None
    assert f"planner failed {MAX_CONSECUTIVE_PLANNER_FAILURES} cycles in a row" in reason
    assert "planner exploded" in reason
    assert world.planner_calls == MAX_CONSECUTIVE_PLANNER_FAILURES
    assert orch.cycles == MAX_CONSECUTIVE_PLANNER_FAILURES
    assert world.tester_calls == 0
    assert orch.test_results == []
    assert world.constructed == []  # no workers ever ran
    errors = [payload for name, payload in events if name == "planner_error"]
    assert len(errors) == MAX_CONSECUTIVE_PLANNER_FAILURES
    assert [payload["consecutive"] for payload in errors] == [1, 2, 3]
    # PRE-FIX behavior: the tester ran on cycles 1 and 2, costing two full suite
    # runs for cycles that produced no plan. The fix introduced _NO_PLAN sentinel
    # and _skip_cycle() so execution and testing are skipped entirely.
    assert "test" not in [name for name, _ in events]


async def test_planner_failure_then_recovery_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tolerated planner failure skips the cycle; a subsequent good plan completes."""
    events: list[tuple[str, dict[str, Any]]] = []
    # Cycle 1: planner fails (tolerated) -> _NO_PLAN -> skip cycle
    # Cycle 2: planner returns a valid plan -> execute -> test passes -> complete
    world = World(
        plans=[RuntimeError("planner exploded"), plan(task("a"))],
        tester_results=[passing()],  # only used on cycle 2
    )
    install_fakes(monkeypatch, world)
    orch = build(
        GOAL,
        world,
        settings=make_settings_for(max_cycles=0),
        on_event=record(events),
    )

    assert await orch.run() is None

    assert orch.stopped_reason == COMPLETE_REASON
    assert orch.cycles == 2
    assert world.planner_calls == 2
    assert world.tester_calls == 1  # tester only ran on cycle 2
    assert world.constructed == ["a"]  # worker only built on cycle 2
    assert orch.tasks[0].status is TaskStatus.COMPLETED

    event_names = [name for name, _ in events]
    # cycle 1: cycle_start, planner_error, cycle_end
    # cycle 2: cycle_start, plan, task_start, task_done, test, cycle_end, halt
    assert event_names == [
        "cycle_start",
        "planner_error",
        "cycle_end",
        "cycle_start",
        "plan",
        "task_start",
        "task_done",
        "test",
        "cycle_end",
        "halt",
    ]
    # planner_error is the second event (index 1)
    assert events[1][1]["consecutive"] == 1  # planner_error on cycle 1
    halt_payload = events[-1][1]
    assert halt_payload["complete"] is True


async def test_unexpected_cycle_error_halts_and_emits_halt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected exception in _cycle is caught, halts with reason, emits halt."""
    events: list[tuple[str, dict[str, Any]]] = []
    world = World(plans=[plan(task("a"))], tester_results=[passing()])
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, on_event=record(events))

    # Monkeypatch _cycle to raise an unexpected error
    async def boom_cycle() -> str | None:
        raise RuntimeError("boom from _cycle")

    monkeypatch.setattr(orch, "_cycle", boom_cycle)

    assert await orch.run() is None  # does not propagate

    event_names = [name for name, _ in events]
    assert event_names == ["cycle_start", "halt"]

    halt_payload = events[1][1]  # the halt payload
    assert halt_payload["complete"] is False
    reason = halt_payload["reason"]
    assert "unexpected orchestrator error" in reason
    assert "RuntimeError" in reason


# --------------------------------------------------------------------------- #
# 12. Workers that do not deliver
# --------------------------------------------------------------------------- #


async def test_worker_with_no_changed_files_is_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason_text = "the model request failed: ReadTimeout"
    world = World(
        plans=[plan(task("a"))],
        worker_scripts={"a": WorkerScript(files=(), error=reason_text)},
        tester_results=[failing("tests/test_a.py::test_a FAILED")],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=1))

    await orch.run()

    assert orch.tasks[0].status is TaskStatus.FAILED
    assert orch.tasks[0].result is not None
    assert reason_text in orch.tasks[0].result  # worker.error is the reason
    assert orch.changed_files == []
    assert world.order == ["start:a", "done:a"]  # it ran, it just changed nothing


async def test_raising_worker_is_marked_failed_distinctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(
        plans=[plan(task("boom"), task("fine"))],
        worker_scripts={"boom": WorkerScript(boom=ValueError("bad wire format"))},
        tester_results=[passing()],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=1))

    await orch.run()

    tasks = by_id(orch.tasks)
    assert tasks["boom"].status is TaskStatus.FAILED
    assert tasks["boom"].result == "worker raised ValueError: bad wire format"
    assert "dependency" not in (tasks["boom"].result or "")  # not the skipped path
    assert "no edit" not in (tasks["boom"].result or "")  # not the no-op path
    assert tasks["fine"].status is TaskStatus.COMPLETED  # one task cannot sink the cycle
    assert world.tester_calls == 1
    reason = orch.stopped_reason
    assert reason == "reached the max_cycles limit of 1 cycle(s)"
    assert not reason.startswith(COMPLETE_PREFIX)


# --------------------------------------------------------------------------- #
# 13. Concurrency cap
# --------------------------------------------------------------------------- #


async def test_concurrency_never_exceeds_max_parallel_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(
        plans=[plan(*(task(f"t{index}") for index in range(10)))],
        tester_results=[passing()],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_parallel_tasks=3))

    await orch.run()

    assert world.peak_in_flight == 3
    assert world.constructed == [f"t{index}" for index in range(10)]
    assert len(world.order) == 20  # every task started and finished
    assert [item.status for item in orch.tasks] == [TaskStatus.COMPLETED] * 10
    assert orch.stopped_reason == COMPLETE_REASON


# --------------------------------------------------------------------------- #
# 14. Progress events
# --------------------------------------------------------------------------- #


async def test_on_event_receives_expected_events(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    world = World(plans=[plan(task("a"))], tester_results=[passing()])
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, on_event=record(events))

    await orch.run()

    assert [name for name, _ in events] == [
        "cycle_start",
        "plan",
        "task_start",
        "task_done",
        "test",
        "cycle_end",
        "halt",
    ]
    payloads = dict(events)
    assert payloads["cycle_start"] == {"cycle": 1, "focus": GOAL}
    assert payloads["plan"]["tasks"] == ["[a] implement a"]
    assert payloads["task_start"] == {"cycle": 1, "task": "a"}
    assert payloads["task_done"]["status"] == TaskStatus.COMPLETED.value
    assert payloads["task_done"]["skipped"] is False
    assert payloads["test"]["passed"] is True
    assert payloads["test"]["verified"] is True
    assert payloads["cycle_end"]["cycle"] == 1
    assert payloads["cycle_end"]["changed_files"] == ["src/a.py"]
    assert payloads["halt"]["complete"] is True
    assert payloads["halt"]["cycles"] == 1


async def test_raising_on_event_hook_never_breaks_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(name: str, payload: dict[str, Any]) -> None:
        raise RuntimeError(f"progress hook failed on {name}: {sorted(payload)}")

    world = World(plans=[plan(task("a"))], tester_results=[passing()])
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, on_event=boom)

    await orch.run()

    assert orch.cycles == 1
    assert orch.stopped_reason == COMPLETE_REASON
    assert [item.status for item in orch.tasks] == [TaskStatus.COMPLETED]


# --------------------------------------------------------------------------- #
# 15. Cancellation
# --------------------------------------------------------------------------- #


async def test_cancellation_records_reason_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(
        plans=[plan(task("a"))],
        tester_results=[passing()],
        tester_gate=asyncio.Event(),  # parks the cycle inside the tester
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=0))

    runner = asyncio.create_task(orch.run())
    await asyncio.wait_for(world.tester_entered.wait(), NO_HANG_TIMEOUT)
    runner.cancel()

    with pytest.raises(asyncio.CancelledError):
        await runner

    assert runner.cancelled()
    assert orch.stopped_reason == "cancelled before the goal was met"
    assert not orch.stopped_reason.startswith(COMPLETE_PREFIX)
    assert orch.cycles == 1
    assert world.tester_goals == [GOAL]  # the goal reached the tester unmolested


# --------------------------------------------------------------------------- #
# 16-17. Empty plan, independent tasks
# --------------------------------------------------------------------------- #


async def test_planner_returning_no_tasks_halts(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(
        plans=[Plan(tasks=[], rationale="nothing left to do")],
        tester_results=[passing()],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=0))

    await asyncio.wait_for(orch.run(), NO_HANG_TIMEOUT)

    assert orch.stopped_reason == "the planner returned no tasks to execute"
    assert not orch.stopped_reason.startswith(COMPLETE_PREFIX)
    assert orch.cycles == 1
    assert orch.tasks == []
    assert world.constructed == []
    assert world.tester_calls == 0


async def test_tasks_without_dependencies_all_run_in_one_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = ["alpha", "beta", "gamma", "delta"]
    world = World(
        plans=[plan(*(task(task_id) for task_id in ids))],
        tester_results=[passing()],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_parallel_tasks=4))

    await orch.run()

    assert (orch.cycles, world.planner_calls, world.tester_calls) == (1, 1, 1)
    assert world.order == [f"start:{task_id}" for task_id in ids] + [
        f"done:{task_id}" for task_id in ids
    ]
    assert world.peak_in_flight == 4  # all four overlapped inside one level
    assert [item.status for item in orch.tasks] == [TaskStatus.COMPLETED] * 4
    assert orch.stopped_reason == COMPLETE_REASON


# --------------------------------------------------------------------------- #
# 18. Repair-context wiring: which cycles hand ``failures`` to the worker
# --------------------------------------------------------------------------- #


async def test_first_cycle_is_not_a_repair_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(plans=[plan(task("a"))], tester_results=[passing()])
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=0))

    await asyncio.wait_for(orch.run(), NO_HANG_TIMEOUT)

    # No prior verdict existed, so the keyword was omitted -- not passed empty.
    assert world.worker_constructions == [("a", None)]
    assert orch.stopped_reason == COMPLETE_REASON


async def test_cycle_after_a_red_verdict_passes_the_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = "tests/test_toml.py::test_nested_table FAILED"
    world = World(
        plans=[plan(task("a")) for _ in range(3)],
        tester_results=[failing(failure), failing(failure), passing()],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=0))

    await asyncio.wait_for(orch.run(), NO_HANG_TIMEOUT)

    assert orch.cycles == 3
    # Cycle 1 has no verdict; cycles 2 and 3 follow a red verified one.
    assert [failures for _, failures in world.worker_constructions] == [
        None,
        (failure,),
        (failure,),
    ]


async def test_cycle_after_a_green_verdict_does_not_pass_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cycle 1 returns a green, verified verdict but does not complete the goal, so
    the loop keeps going; cycle 2 (which followed a green verdict) still gets no
    repair context.

    Cycle 1's worker is scripted to apply no edits, so its task is marked FAILED
    and ``_is_done()`` stays false despite the green suite -- the bug this test
    guards against is the original "exit on the first green" behaviour, which
    never builds a second worker.
    """
    world = World(
        plans=[plan(task("a")) for _ in range(3)],
        worker_scripts={"a": WorkerScript(files=())},
        tester_results=[passing(), passing()],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=2))

    await asyncio.wait_for(orch.run(), NO_HANG_TIMEOUT)

    # Cycle 1 had no prior verdict and cycle 2 followed a green one: no context
    # in either, so the plain prompt both times.
    assert [failures for _, failures in world.worker_constructions] == [None, None]


async def test_unverified_verdict_does_not_trigger_a_repair_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(
        plans=[plan(task("a"))],
        tester_results=[unverified()],
    )
    install_fakes(monkeypatch, world)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=0))

    await asyncio.wait_for(orch.run(), NO_HANG_TIMEOUT)

    # The unverifiable verdict halts the loop outright rather than repairing.
    assert orch.cycles == 1
    assert "unverifiable, not passing" in orch.stopped_reason
    assert world.worker_constructions == [("a", None)]
    # And with that verdict still the last one on record, a worker built for a
    # further cycle would still get no repair context -- ``verified=False`` is
    # never a repair signal, whatever ``passed`` says.
    assert orch._repair_failures() == []
    before = len(world.worker_constructions)
    orch._build_worker(task("a"))
    assert world.worker_constructions[before:] == [("a", None)]


async def test_worker_constructed_without_failures_when_no_repair_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-context path is the plain historical call, kwargs and all."""
    from furrow.core import orchestrator as orchestrator_module

    world = World(plans=[plan(task("a"))], tester_results=[passing()])
    install_fakes(monkeypatch, world)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    installed = orchestrator_module.WorkerAgent

    class SpyWorker:
        """``installed`` verbatim, recording the exact positional/kw call."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))
            self._worker = installed(*args, **kwargs)

        async def run(self) -> str:
            return await self._worker.run()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._worker, name)

    monkeypatch.setattr(orchestrator_module, "WorkerAgent", SpyWorker)
    orch = build(GOAL, world, settings=make_settings_for(max_cycles=0))

    await asyncio.wait_for(orch.run(), NO_HANG_TIMEOUT)

    assert len(calls) == 1
    args, kwargs = calls[0]
    # Exactly ``WorkerAgent(task=..., client=...)``: no positional third argument,
    # no ``settings``, and above all no ``failures`` keyword to fall back from.
    assert args == ()
    assert kwargs == {"task": orch.tasks[0], "client": orch.client}
    assert "failures" not in kwargs
    assert world.worker_constructions == [("a", None)]
