"""Tests for the run loop and the event bus: ``furrow/core/events.py`` + ``orchestrator.py``.

Two halves, because they are two halves of one contract:

* **events** -- the wire format the web and CLI front-ends both consume. The first test in
  this file (:func:`test_emit_log_type_is_the_wire_string_not_the_enum_repr`) is the
  regression test for a real, shipped bug: for a ``(str, Enum)`` mixin ``str()`` is
  version-dependent, and on Python 3.10 ``str(EventType.LOG)`` yields ``"EventType.LOG"``
  rather than ``"log"``. That string is not a valid wire value, so every
  ``== EventType.X.value`` comparison downstream fails -- including the dispatch inside
  :func:`furrow.core.events._render_event`, which meant the console mirror silently
  rendered nothing at all. The rest of the events section pins the other properties the
  web socket handler relies on: every event is ``json.dumps``-able whatever it carries,
  ``cycle`` is auto-stamped, and a subscriber that explodes cannot take the run down.
* **the loop** -- :func:`test_two_cycles_run_when_the_first_test_fails_and_the_second_passes`
  is the headline regression test. ``_get_tasks()`` used to return ``[]`` unconditionally,
  which made ``_is_done()`` see ``completed == 0`` and ``0 >= 0`` and stop after a single
  cycle: the run printed the failures and then immediately printed "Goal complete.
  Halting." The rest of the section covers the stop conditions, the original-goal
  immutability, dependency scheduling, the concurrency cap and the event payloads the
  front-ends parse.

Everything here is offline. The planner, tester and worker are hand-rolled fakes defined
**locally in this module** (they are not shared fixtures: only this file needs them, and
``conftest.py`` belongs to the suite as a whole). They are injected the way production
injects them -- ``orch.planner = ...`` / ``orch.tester = ...`` attribute assignment, and
``monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", ...)`` for the worker,
because ``_run_task`` resolves that name at call time.

The hermetic ``settings`` fixture from ``conftest.py`` is the base for every
:class:`~furrow.config.Settings` here: it already points ``workspace`` at a temp
directory and carries a dummy API key, so no test can read or write the checkout.
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from furrow.config import Settings
from furrow.core.events import Event, EventEmitter, EventType
from furrow.core.orchestrator import DEFAULT_NO_PROGRESS_LIMIT, Orchestrator
from furrow.models import Plan, TaskModel, TaskStatus

# Aliased: a bare `TestResult` name in a test module trips pytest's `Test*`
# class-collection heuristic and emits a PytestCollectionWarning.
from furrow.models import TestResult as Verdict

#: The original user requirement. Every loop test asserts it survives verbatim.
GOAL = "make the thing work"


# --------------------------------------------------------------------------- #
# Local helpers and fakes
# --------------------------------------------------------------------------- #


def make_task(
    ident: str,
    description: str | None = None,
    *,
    dependencies: list[str] | None = None,
    files: list[str] | None = None,
) -> TaskModel:
    """Build one :class:`~furrow.models.TaskModel` with sensible defaults."""
    return TaskModel(
        id=ident,
        description=description or f"do {ident}",
        dependencies=list(dependencies or []),
        files=list(files or []),
    )


def make_plan(*tasks: TaskModel, rationale: str = "because") -> Plan:
    """Build a :class:`~furrow.models.Plan` around `tasks`."""
    return Plan(rationale=rationale, tasks=list(tasks))


def make_verdict(
    *,
    passed: bool = False,
    skipped: bool = False,
    summary: str = "",
    failures: list[str] | None = None,
) -> Verdict:
    """Build a :class:`~furrow.models.TestResult`."""
    return Verdict(
        passed=passed,
        skipped=skipped,
        summary=summary,
        failures=list(failures or []),
    )


class FakePlanner:
    """Replays a scripted list of plans (or exceptions) and records every goal.

    The last entry repeats once the script is exhausted, so a test only has to spell out
    as many cycles as it cares about.
    """

    def __init__(self, script: list[Plan | BaseException]) -> None:
        """See class docstring."""
        self.script = list(script)
        self.goals: list[str] = []

    async def plan(self, goal: str) -> Plan:
        """Return this cycle's scripted plan, or raise the scripted exception."""
        self.goals.append(goal)
        item = self.script[min(len(self.goals) - 1, len(self.script) - 1)]
        if isinstance(item, BaseException):
            raise item
        # Hand back a deep-ish copy so a cycle's mutation never leaks into the script.
        return item.model_copy(deep=True)


class CountingPlanner:
    """Synthesises one fresh task per cycle, so no cycle is ever a no-op dedup."""

    def __init__(self) -> None:
        """See class docstring."""
        self.goals: list[str] = []

    async def plan(self, goal: str) -> Plan:
        """Return a one-task plan whose id is unique to this cycle."""
        self.goals.append(goal)
        return make_plan(make_task(f"t{len(self.goals)}", f"work number {len(self.goals)}"))


class FixedPlanner:
    """Returns the same plan on every cycle, so only the dedup path is under test."""

    def __init__(self, *tasks: TaskModel) -> None:
        """See class docstring."""
        self._plan = make_plan(*tasks)
        self.goals: list[str] = []

    async def plan(self, goal: str) -> Plan:
        """Return a fresh copy of the fixed plan."""
        self.goals.append(goal)
        return self._plan.model_copy(deep=True)


class FakeTester:
    """Replays a scripted list of verdicts (or exceptions), recording each call."""

    def __init__(self, script: list[Verdict | BaseException]) -> None:
        """See class docstring."""
        self.script = list(script)
        self.calls: list[tuple[str, list[str]]] = []

    async def run(self, goal: str, tasks: list[TaskModel]) -> Verdict:
        """Return this cycle's scripted verdict, or raise the scripted exception."""
        self.calls.append((goal, [t.id for t in tasks]))
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item.model_copy(deep=True)


class FakeWorker:
    """Stands in for :class:`~furrow.agents.worker.WorkerAgent`.

    ``handler(worker)`` decides what happens: it may raise (to model a failing task),
    populate ``worker.written_files`` (to model progress) or record its own timings.
    """

    def __init__(self, task_model: TaskModel, handler: Any) -> None:
        """See class docstring."""
        self.task = task_model
        self._handler = handler
        self.written_files: list[str] = []
        self.errors: list[str] = []

    async def run(self) -> str:
        """Run the handler and return the canned summary for this task."""
        if self._handler is not None:
            await self._handler(self)
        return f"summary for {self.task.id}"


def worker_factory(
    handler: Any = None,
    *,
    log: list[str] | None = None,
    writes: list[str] | None = None,
) -> Any:
    """Build a drop-in replacement for ``furrow.core.orchestrator.WorkerAgent``.

    Returns a callable accepting the same arguments as the real constructor, so it can be
    handed straight to :func:`pytest.MonkeyPatch.setattr`.
    """

    async def _run(worker: FakeWorker) -> None:
        if writes is not None:
            worker.written_files.extend(writes)
        if handler is not None:
            result = handler(worker)
            if asyncio.iscoroutine(result):
                await result

    def factory(*args: Any, **kwargs: Any) -> FakeWorker:
        task_model: TaskModel = kwargs.get("task") or args[0]
        if log is not None:
            log.append(task_model.id)
        return FakeWorker(task_model, _run)

    return factory


class Ticker:
    """Monotonic sequence counter plus a live-concurrency gauge."""

    def __init__(self) -> None:
        """See class docstring."""
        self.seq = 0
        self.live = 0
        self.peak = 0
        self.started: dict[str, int] = {}
        self.stopped: dict[str, int] = {}
        self.order: list[str] = []

    def enter(self, ident: str) -> None:
        """Mark `ident` as started and bump the live gauge."""
        self.seq += 1
        self.started[ident] = self.seq
        self.order.append(ident)
        self.live += 1
        self.peak = max(self.peak, self.live)

    def leave(self, ident: str) -> None:
        """Mark `ident` as finished and drop the live gauge."""
        self.seq += 1
        self.stopped[ident] = self.seq
        self.live -= 1


@pytest.fixture
def settings_factory(settings: Settings) -> Callable[..., Settings]:
    """Derive per-test :class:`~furrow.config.Settings` variants from the shared fixture.

    Built with ``model_copy(update=...)`` so the hermetic fixture stays the single source
    of truth for ``workspace`` and the dummy credential, and a test only has to name the
    one or two knobs it actually cares about (``max_cycles``, ``max_parallel_tasks``).
    """

    def build(**overrides: Any) -> Settings:
        """Return the fixture's settings with `overrides` applied."""
        return settings.model_copy(update=overrides)

    return build


def build_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator_settings: Settings,
    *,
    planner: Any,
    tester: Any,
    handler: Any = None,
    worker_log: list[str] | None = None,
    writes: list[str] | None = None,
    on_event: Any = None,
    goal: str = GOAL,
) -> Orchestrator:
    """Assemble an orchestrator with every real agent replaced by a fake."""
    monkeypatch.setattr(
        "furrow.core.orchestrator.WorkerAgent",
        worker_factory(handler, log=worker_log, writes=writes),
    )
    orch = Orchestrator(
        goal,
        client=object(),  # type: ignore[arg-type]
        settings=orchestrator_settings,
        on_event=on_event,
    )
    orch.planner = planner
    orch.tester = tester
    return orch


def types_of(events: list[Event]) -> list[str]:
    """The ordered ``type`` values of `events`."""
    return [event["type"] for event in events]


# --------------------------------------------------------------------------- #
# events.py
# --------------------------------------------------------------------------- #


def test_emit_log_type_is_the_wire_string_not_the_enum_repr() -> None:
    """``emit(EventType.LOG)["type"]`` is ``"log"``.

    This is the regression test for the ``str()``-on-a-``(str, Enum)`` bug: on Python 3.10
    ``str(EventType.LOG)`` is ``"EventType.LOG"``, which broke every dispatch that compared
    the event type against a member value -- including the console render -- while looking
    perfectly healthy.
    """
    emitter = EventEmitter()
    event = emitter.emit(EventType.LOG, message="hi")
    assert event["type"] == "log"
    assert type(event["type"]) is str


@pytest.mark.parametrize("member", list(EventType), ids=lambda m: m.value)
def test_emit_type_is_member_value_for_every_event_type(member: EventType) -> None:
    """For every member, the type field is the member's ``.value`` -- never its repr."""
    event = EventEmitter().emit(member, message="payload")
    assert event["type"] == member.value
    assert not isinstance(event["type"], Enum)
    assert f"EventType.{member.name}" not in json.dumps(event)


def test_emit_accepts_an_arbitrary_string_type_verbatim() -> None:
    """A caller-supplied string is preserved, not coerced to a member."""
    event = EventEmitter().emit("custom_event", payload=1)
    assert event["type"] == "custom_event"
    assert type(event["type"]) is str


def test_every_event_is_json_dumps_able_despite_awkward_fields() -> None:
    """``Path`` / ``TaskModel`` / ``TaskStatus`` / nested containers all survive the wire."""
    emitter = EventEmitter()
    event = emitter.emit(
        EventType.PLAN,
        where=Path("/tmp/example.py"),
        task=make_task("t1", "write it", files=["a.py"]),
        status=TaskStatus.COMPLETED,
        nested={"deps": ["a", "b"], "inner": {"status": TaskStatus.FAILED}},
        listed=[TaskStatus.PENDING, Path("rel/x.py"), {"k": TaskStatus.SKIPPED}],
    )
    payload = json.dumps(event)
    restored = json.loads(payload)
    assert restored["where"] == "/tmp/example.py"
    assert restored["task"]["id"] == "t1"
    assert restored["status"] == "completed"
    assert restored["nested"] == {"deps": ["a", "b"], "inner": {"status": "failed"}}
    assert restored["listed"] == ["pending", "rel/x.py", {"k": "skipped"}]


def test_current_cycle_is_stamped_and_an_explicit_cycle_overrides_it() -> None:
    """The ``cycle`` key is auto-filled from ``current_cycle`` unless passed explicitly."""
    emitter = EventEmitter()
    assert emitter.emit(EventType.LOG, message="a")["cycle"] == 0
    emitter.current_cycle = 4
    assert emitter.emit(EventType.LOG, message="b")["cycle"] == 4
    assert emitter.emit(EventType.LOG, message="c", cycle=9)["cycle"] == 9


def test_a_raising_subscriber_neither_propagates_nor_blocks_later_ones() -> None:
    """One bad listener is quarantined: no propagation, no reordering, no lost event."""
    seen: list[Event] = []

    def boom(_event: Event) -> None:
        raise RuntimeError("listener exploded")

    emitter = EventEmitter()
    emitter.subscribe(boom)
    emitter.subscribe(seen.append)
    event = emitter.emit(EventType.LOG, message="still delivered")
    assert seen == [event]
    assert emitter.history == [event]


def test_a_subscriber_may_unsubscribe_itself_during_fan_out() -> None:
    """Mutating the subscriber list mid-emit does not skip anyone or double-deliver."""
    seen: list[str] = []
    emitter = EventEmitter()
    unsubscribe: Callable[[], None]

    def leaky(_event: Event) -> None:
        seen.append("self-removing")
        unsubscribe()

    unsubscribe = emitter.subscribe(leaky)
    emitter.subscribe(lambda _event: seen.append("later"))
    emitter.emit(EventType.LOG, message="one")
    emitter.emit(EventType.LOG, message="two")
    assert seen == ["self-removing", "later", "later"]
    assert len(emitter.history) == 2


def test_subscribe_returns_a_working_unsubscribe() -> None:
    """The callable returned by ``subscribe`` really detaches the callback, idempotently."""
    seen: list[Event] = []
    emitter = EventEmitter()
    unsubscribe = emitter.subscribe(seen.append)
    emitter.emit(EventType.LOG, message="before")
    unsubscribe()
    unsubscribe()
    emitter.emit(EventType.LOG, message="after")
    assert len(seen) == 1
    assert seen[0]["message"] == "before"


_RENDER_CASES: list[tuple[EventType, dict[str, Any], str]] = [
    (EventType.RUN_START, {"goal": "ship it"}, "ship it"),
    (EventType.CYCLE_START, {}, "cycle start"),
    (EventType.PLAN, {"task_count": 2, "rationale": "because reasons"}, "2 tasks"),
    (EventType.TASK_START, {"task_id": "t1", "description": "write a.py"}, "write a.py"),
    (
        EventType.TASK_END,
        {"task_id": "t1", "status": "completed", "written_files": ["a.py", "b.py"]},
        "2 writes",
    ),
    (
        EventType.TEST,
        {"passed": False, "skipped": False, "summary": "3 failed"},
        "3 failed",
    ),
    (
        EventType.CYCLE_END,
        {"completed": 1, "failed": 2, "skipped": 3, "passed": False},
        "2 fail",
    ),
    (EventType.RUN_END, {"cycles": 4, "stopped_because": "max-cycles"}, "max-cycles"),
    (EventType.LOG, {"message": "hello there"}, "hello there"),
    (EventType.ERROR, {"stage": "plan", "message": "it broke"}, "it broke"),
]


@pytest.mark.parametrize(
    ("member", "fields", "expected_body"),
    _RENDER_CASES,
    ids=[case[0].value for case in _RENDER_CASES],
)
def test_render_event_renders_a_body_for_every_event_type(
    member: EventType, fields: dict[str, Any], expected_body: str
) -> None:
    """Every member reaches a dispatch branch that produces a real body.

    The second assertion is the regression: when the type carried ``"EventType.X"``
    instead of the member value, *every* comparison in the dispatch fell through to
    ``body = etype`` and the console printed ``(cycle 0) log: log``.
    """
    buffer = io.StringIO()
    EventEmitter(console=Console(file=buffer, width=200)).emit(member, **fields)
    out = buffer.getvalue()
    assert out.strip(), "nothing was rendered at all"
    assert f"(cycle 0) {member.value}: " in out
    assert expected_body in out
    assert out.strip() != f"(cycle 0) {member.value}: {member.value}"


def test_a_message_containing_rich_markup_is_escaped_not_interpreted() -> None:
    """``[bold]`` in LLM prose renders literally instead of raising ``MarkupError``."""
    buffer = io.StringIO()
    emitter = EventEmitter(console=Console(file=buffer, width=200))
    event = emitter.emit(EventType.LOG, message="[bold]oops[/] and [red]bad[/]")
    out = buffer.getvalue()
    assert "[bold]oops[/]" in out
    assert event["message"] == "[bold]oops[/] and [red]bad[/]"


def test_keep_history_false_leaves_history_empty_but_still_delivers() -> None:
    """Retention is independent of delivery: subscribers still get everything."""
    seen: list[Event] = []
    emitter = EventEmitter(seen.append, keep_history=False)
    emitter.emit(EventType.LOG, message="live")
    emitter.emit(EventType.ERROR, stage="plan", message="boom")
    assert len(seen) == 2
    assert emitter.history == []
    emitter.clear()
    assert emitter.history == []


def test_clear_empties_history_but_leaves_subscribers_attached() -> None:
    """``run()`` calls ``clear()`` on every start, so a re-run must not re-deliver."""
    seen: list[Event] = []
    emitter = EventEmitter(seen.append)
    emitter.emit(EventType.LOG, message="first run")
    assert len(emitter.history) == 1
    emitter.clear()
    assert emitter.history == []
    emitter.emit(EventType.LOG, message="second run")
    assert [event["message"] for event in seen] == ["first run", "second run"]


# --------------------------------------------------------------------------- #
# orchestrator -- the loop
# --------------------------------------------------------------------------- #


async def test_two_cycles_run_when_the_first_test_fails_and_the_second_passes(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """The headline regression: a failing cycle must not end the run.

    ``_get_tasks()`` used to return ``[]`` unconditionally, so ``_is_done()`` evaluated
    ``0 >= 0`` and stopped after cycle 1 -- printing the failures and then "Goal complete.
    Halting." in the same breath.
    """
    planner = FakePlanner([make_plan(make_task("t1", "first attempt"))])
    tester = FakeTester(
        [
            make_verdict(passed=False, summary="1 failed", failures=["assert 1 == 2"]),
            make_verdict(passed=True, summary="all green"),
        ]
    )
    orch = build_orchestrator(
        monkeypatch, settings_factory(), planner=planner, tester=tester
    )

    summary = await orch.run()

    assert summary.cycles == 2
    assert summary.stopped_because == "tests-passed"
    assert len(planner.goals) == 2
    assert len(tester.calls) == 2


async def test_a_passing_test_on_cycle_one_stops_after_exactly_one_cycle(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """A pass is the natural terminal condition, not a floor on the cycle count."""
    tester = FakeTester([make_verdict(passed=True, summary="all green")])
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=CountingPlanner(),
        tester=tester,
    )

    summary = await orch.run()

    assert summary.cycles == 1
    assert summary.stopped_because == "tests-passed"
    assert len(tester.calls) == 1


async def test_a_failing_test_does_not_stop_the_loop(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """Two failures in a row must not end the run -- only a pass does."""
    tester = FakeTester(
        [
            make_verdict(passed=False, failures=["fail one"]),
            make_verdict(passed=False, failures=["fail two"]),
            make_verdict(passed=True, summary="finally"),
        ]
    )
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(max_cycles=9),
        planner=CountingPlanner(),
        tester=tester,
    )

    summary = await orch.run()

    assert summary.cycles == 3
    assert summary.stopped_because == "tests-passed"


async def test_max_cycles_two_runs_exactly_two_cycles(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """``max_cycles`` is a hard cap and reports itself as the reason."""
    tester = FakeTester([make_verdict(passed=False, failures=["still red"])])
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(max_cycles=2),
        planner=CountingPlanner(),
        tester=tester,
    )

    summary = await orch.run()

    assert summary.cycles == 2
    assert summary.stopped_because == "max-cycles"
    assert len(tester.calls) == 2


async def test_a_worker_that_never_writes_stops_on_the_no_progress_limit(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """``max_cycles=0`` must not mean "spin forever rewriting an empty plan"."""
    tester = FakeTester([make_verdict(passed=False, failures=["nothing landed"])])
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(max_cycles=0),
        planner=CountingPlanner(),
        tester=tester,
    )

    summary = await orch.run()

    assert summary.stopped_because == "no-progress"
    assert summary.cycles == DEFAULT_NO_PROGRESS_LIMIT


async def test_a_worker_that_writes_resets_the_no_progress_counter(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """Cycles 1 and 3 write, so the run must get past the limit and stop at cycle 6."""
    writing = {"t1", "t3"}
    ticker = Ticker()

    def handler(worker: Any) -> None:
        ticker.enter(worker.task.id)
        if worker.task.id in writing:
            worker.written_files.append(f"{worker.task.id}.py")
        ticker.leave(worker.task.id)

    tester = FakeTester([make_verdict(passed=False, failures=["still red"])])
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(max_cycles=0),
        planner=CountingPlanner(),
        tester=tester,
        handler=handler,
    )

    summary = await orch.run()

    assert sorted(ticker.order) == [f"t{n}" for n in range(1, 7)]
    assert summary.cycles == 6
    assert summary.stopped_because == "no-progress"


async def test_the_original_goal_survives_and_failures_are_appended(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """``self.goal`` is immutable; context is appended to the *planning* goal only.

    The old code did ``self.goal = "Fix failing tests:\\n..."`` on every failure, so by the
    third cycle the user's original requirement had been overwritten and forgotten.
    """
    tester = FakeTester(
        [
            make_verdict(passed=False, failures=["assert 1 == 2"]),
            make_verdict(passed=True, summary="green"),
        ]
    )
    planner = CountingPlanner()
    orch = build_orchestrator(
        monkeypatch, settings_factory(), planner=planner, tester=tester
    )

    summary = await orch.run()

    assert orch.goal == GOAL
    assert summary.goal == GOAL
    assert planner.goals[0] == GOAL
    assert planner.goals[1].startswith(GOAL)
    assert planner.goals[1] != GOAL
    assert "assert 1 == 2" in planner.goals[1]
    # The tester always sees the untouched goal, never the failure appendix.
    assert [goal for goal, _ in tester.calls] == [GOAL, GOAL]


async def test_a_skipped_test_result_neither_stops_the_loop_nor_counts_as_a_pass(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """``skipped`` is neutral: the loop keeps going and a LOG explains why."""
    tester = FakeTester(
        [
            make_verdict(passed=True, skipped=True, summary="No runnable test suite detected"),
            make_verdict(passed=True, skipped=True, summary="No runnable test suite detected"),
        ]
    )
    events: list[Event] = []
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(max_cycles=2),
        planner=CountingPlanner(),
        tester=tester,
        on_event=events.append,
    )

    summary = await orch.run()

    assert summary.cycles == 2
    assert summary.stopped_because == "max-cycles"
    logs = [e for e in events if e["type"] == "log"]
    assert any("no runnable test suite" in e["message"] for e in logs)
    test_events = [e for e in events if e["type"] == "test"]
    assert test_events
    assert all(e["skipped"] is True for e in test_events)


async def test_a_planner_value_error_stops_the_run_cleanly(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """An unparseable planner reply is an ``ERROR`` event, not a traceback."""
    planner = FakePlanner([ValueError("the planner returned no tasks for this goal")])
    events: list[Event] = []
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=planner,
        tester=FakeTester([make_verdict(passed=True)]),
        on_event=events.append,
    )

    summary = await orch.run()  # must not raise

    assert summary.stopped_because == "planner-error"
    assert summary.cycles == 1
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["stage"] == "plan"
    assert "no tasks" in errors[0]["message"]


async def test_dependencies_are_respected_and_never_overlap(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """a -> b -> c executes strictly in order, one wave at a time."""
    ticker = Ticker()

    async def handler(worker: Any) -> None:
        ticker.enter(worker.task.id)
        await asyncio.sleep(0)
        ticker.leave(worker.task.id)

    orch = build_orchestrator(
        monkeypatch,
        settings_factory(max_parallel_tasks=4),
        planner=FakePlanner(
            [
                make_plan(
                    make_task("a", "first"),
                    make_task("b", "second", dependencies=["a"]),
                    make_task("c", "third", dependencies=["b"]),
                )
            ]
        ),
        tester=FakeTester([make_verdict(passed=True, summary="green")]),
        handler=handler,
    )

    summary = await orch.run()

    assert ticker.order == ["a", "b", "c"]
    assert ticker.stopped["a"] < ticker.started["b"]
    assert ticker.stopped["b"] < ticker.started["c"]
    assert ticker.peak == 1
    assert summary.completed == 3
    assert summary.failed == 0
    assert summary.skipped == 0


async def test_a_task_blocked_by_a_failed_dependency_is_skipped_not_run(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """A failed dependency skips its dependents instead of running them anyway."""

    async def handler(worker: Any) -> None:
        if worker.task.id == "a":
            raise ValueError("the model asked for an unwritable path")

    log: list[str] = []
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=FakePlanner(
            [make_plan(make_task("a", "fails"), make_task("b", "blocked", dependencies=["a"]))]
        ),
        tester=FakeTester([make_verdict(passed=True, summary="green")]),
        handler=handler,
        worker_log=log,
    )

    summary = await orch.run()

    assert log == ["a"]
    by_id = {t.id: t for t in summary.tasks}
    assert by_id["a"].status == TaskStatus.FAILED
    assert isinstance(by_id["a"].result, str)
    assert "unwritable path" in by_id["a"].result
    assert by_id["b"].status == TaskStatus.SKIPPED
    assert isinstance(by_id["b"].result, str)
    assert "a" in by_id["b"].result
    assert "dependency" in by_id["b"].result
    assert summary.failed == 1
    assert summary.skipped == 1


async def test_a_dependency_cycle_terminates_and_skips_everything(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """a -> b -> a must not hang the wave scheduler.

    ``asyncio.wait_for`` is the assertion: without the cycle detection this future never
    completes and the test times out instead of failing on an assertion.
    """
    log: list[str] = []
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=FakePlanner(
            [
                make_plan(
                    make_task("a", "a waits on b", dependencies=["b"]),
                    make_task("b", "b waits on a", dependencies=["a"]),
                )
            ]
        ),
        tester=FakeTester([make_verdict(passed=True, summary="green")]),
        worker_log=log,
    )

    summary = await asyncio.wait_for(orch.run(), timeout=5)

    assert log == []
    assert {t.status for t in summary.tasks} == {TaskStatus.SKIPPED}
    assert all(
        isinstance(t.result, str) and "dependency cycle" in t.result for t in summary.tasks
    )
    assert summary.skipped == 2


async def test_max_parallel_tasks_bounds_peak_concurrency(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """The semaphore is real: 6 tasks at a cap of 2 never exceed 2 concurrent workers."""
    ticker = Ticker()

    async def handler(worker: Any) -> None:
        ticker.enter(worker.task.id)
        await asyncio.sleep(0.02)
        worker.written_files.append(f"{worker.task.id}.py")
        ticker.leave(worker.task.id)

    orch = build_orchestrator(
        monkeypatch,
        settings_factory(max_parallel_tasks=2),
        planner=FakePlanner([make_plan(*[make_task(f"t{n}") for n in range(6)])]),
        tester=FakeTester([make_verdict(passed=True, summary="green")]),
        handler=handler,
    )

    summary = await orch.run()

    assert len(ticker.order) == 6
    assert ticker.peak == 2
    assert ticker.live == 0
    assert summary.completed == 6


async def test_a_worker_raising_cancelled_error_is_a_failure_not_a_success(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """``CancelledError`` is a ``BaseException`` and must not be misread as success.

    The old ``isinstance(result, Exception)`` check let it through as a "completed" task
    whose ``result`` was an exception *object* -- which then broke ``model_dump`` and any
    ``result.startswith(...)`` the front-ends attempt.
    """

    async def handler(worker: Any) -> None:
        if worker.task.id == "boom":
            raise asyncio.CancelledError

    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=FakePlanner([make_plan(make_task("boom"), make_task("fine"))]),
        tester=FakeTester([make_verdict(passed=True, summary="green")]),
        handler=handler,
    )

    summary = await orch.run()

    by_id = {t.id: t for t in summary.tasks}
    assert by_id["boom"].status == TaskStatus.FAILED
    assert isinstance(by_id["boom"].result, str)
    assert not isinstance(by_id["boom"].result, asyncio.CancelledError)
    assert by_id["fine"].status == TaskStatus.COMPLETED
    assert summary.failed == 1
    assert summary.completed == 1


async def test_a_worker_raising_value_error_still_runs_the_tester(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """A per-task failure is recorded against that task, not fatal to the batch."""

    async def handler(worker: Any) -> None:
        if worker.task.id == "bad":
            raise ValueError("worker returned output that is not a WorkerResult")

    tester = FakeTester([make_verdict(passed=True, summary="green")])
    events: list[Event] = []
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=FakePlanner([make_plan(make_task("bad"), make_task("good"))]),
        tester=tester,
        handler=handler,
        on_event=events.append,
    )

    summary = await orch.run()

    by_id = {t.id: t for t in summary.tasks}
    assert by_id["bad"].status == TaskStatus.FAILED
    failed_result = by_id["bad"].result
    assert isinstance(failed_result, str)
    assert "WorkerResult" in failed_result
    assert by_id["good"].status == TaskStatus.COMPLETED
    assert len(tester.calls) == 1
    assert tester.calls[0][1] == ["bad", "good"]
    assert [e for e in events if e["type"] == "test"]


async def test_cancelling_run_from_outside_reports_cancelled_and_propagates(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """The ``run_end`` event must name the real reason, and the cancellation propagates."""
    started = asyncio.Event()

    async def handler(_worker: Any) -> None:
        started.set()
        await asyncio.sleep(3600)

    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=CountingPlanner(),
        tester=FakeTester([make_verdict(passed=False, failures=["never reached"])]),
        handler=handler,
    )

    task_handle = asyncio.create_task(orch.run())
    await asyncio.wait_for(started.wait(), timeout=5)
    task_handle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_handle

    run_end = [e for e in orch.emitter.history if e["type"] == "run_end"]
    assert len(run_end) == 1
    assert run_end[0]["stopped_because"] == "cancelled"


async def test_an_unexpected_exception_emits_an_error_event_and_propagates(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """Front-ends need the message on the wire; the caller still needs the exception."""
    events: list[Event] = []
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=CountingPlanner(),
        tester=FakeTester([RuntimeError("the tester exploded")]),
        on_event=events.append,
    )

    with pytest.raises(RuntimeError, match="the tester exploded"):
        await orch.run()

    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["stage"] == "run"
    assert errors[0]["message"] == "the tester exploded"
    assert errors[0]["type"] == "error"
    assert "the tester exploded" in json.dumps(errors[0])


async def test_repeated_tasks_are_deduped_and_results_are_not_clobbered(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """A repeat of ``(id, description)`` must not reset or re-run a completed task."""
    log: list[str] = []
    tester = FakeTester(
        [
            make_verdict(passed=False, failures=["needs a second task"]),
            make_verdict(passed=True, summary="green"),
        ]
    )
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=FakePlanner(
            [
                make_plan(make_task("t1", "do the first thing")),
                make_plan(
                    make_task("t1", "do the first thing"),
                    make_task("t2", "do the second thing"),
                ),
            ]
        ),
        tester=tester,
        handler=lambda worker: worker.written_files.append(f"{worker.task.id}.py"),
        worker_log=log,
    )

    summary = await orch.run()

    assert log == ["t1", "t2"], "t1 must not be executed twice"
    assert [t.id for t in summary.tasks] == ["t1", "t2"]
    assert summary.tasks[0].status == TaskStatus.COMPLETED
    assert summary.tasks[0].result == "summary for t1"
    assert tester.calls[0][1] == ["t1"]
    assert tester.calls[1][1] == ["t2"], "the deduped task is not re-run in cycle 2"
    assert json.dumps(summary.to_dict())


async def test_run_summary_to_dict_is_json_serialisable_with_exact_keys(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """``to_dict()`` is the CLI ``--json`` payload and the web ``summary`` field."""
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=CountingPlanner(),
        tester=FakeTester([make_verdict(passed=True, summary="green")]),
        handler=lambda worker: worker.written_files.append(f"{worker.task.id}.py"),
    )

    summary = await orch.run()
    payload = summary.to_dict()

    assert set(payload) == {
        "goal",
        "cycles",
        "completed",
        "failed",
        "skipped",
        "stopped_because",
        "tasks",
    }
    assert json.loads(json.dumps(payload)) == payload
    assert payload["goal"] == GOAL
    assert payload["stopped_because"] == "tests-passed"
    assert payload["tasks"] == [
        {
            "id": "t1",
            "description": "work number 1",
            "files": [],
            "dependencies": [],
            "status": "completed",
            "result": "summary for t1",
        }
    ]


async def test_rerunning_one_orchestrator_resets_every_piece_of_state(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """``run()`` is re-entrant: a second run sees neither the tasks nor the failures.

    ``run()`` resets ``tasks``, the failure appendix, the no-progress counter and the event
    history on entry; without that, the second run would dedup against cycle 1's leftovers
    and stop after doing nothing.
    """
    tester = FakeTester(
        [
            make_verdict(passed=False, failures=["first run fails"]),
            make_verdict(passed=True, summary="green"),
        ]
    )
    planner = FixedPlanner(make_task("t1", "write a.py"))
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=planner,
        tester=tester,
        handler=lambda worker: worker.written_files.append("a.py"),
    )

    first = await orch.run()
    second = await orch.run()

    assert first.cycles == 2
    assert first.failed == 0
    assert second.cycles == 1
    assert second.stopped_because == "tests-passed"
    assert second.goal == GOAL
    assert [t.id for t in second.tasks] == ["t1"]
    assert [t.status for t in second.tasks] == [TaskStatus.COMPLETED]
    # Cycle 2 of run 1 carried the failure appendix; run 2's first cycle starts clean.
    assert "first run fails" in planner.goals[1]
    assert planner.goals[2] == GOAL
    assert types_of(orch.emitter.history)[0] == "run_start"
    assert orch.emitter.history[-1]["stopped_because"] == "tests-passed"


# --------------------------------------------------------------------------- #
# orchestrator -- events emitted
# --------------------------------------------------------------------------- #


async def test_on_event_sees_the_whole_lifecycle_in_order(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """The callback the web layer subscribes to sees every event, in emission order."""
    events: list[Event] = []
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=FakePlanner([make_plan(make_task("t1", "do it"), make_task("t2", "and this"))]),
        tester=FakeTester([make_verdict(passed=True, summary="green")]),
        handler=lambda worker: worker.written_files.append(f"{worker.task.id}.py"),
        on_event=events.append,
    )

    await orch.run()

    assert types_of(events) == [
        "run_start",
        "cycle_start",
        "plan",
        "task_start",
        "task_start",
        "task_end",
        "task_end",
        "test",
        "cycle_end",
        "run_end",
    ]
    assert events == orch.emitter.history
    assert events[0]["goal"] == GOAL
    assert events[0]["cycle"] == 0
    assert events[-1]["stopped_because"] == "tests-passed"
    plan_event = events[2]
    assert plan_event["task_count"] == 2
    assert [t["id"] for t in plan_event["tasks"]] == ["t1", "t2"]


async def test_task_end_carries_a_plain_string_status(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """An enum repr on the wire breaks every JSON consumer of ``task_end``."""

    async def handler(worker: Any) -> None:
        if worker.task.id == "bad":
            raise ValueError("nope")

    events: list[Event] = []
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=FakePlanner([make_plan(make_task("ok"), make_task("bad"))]),
        tester=FakeTester([make_verdict(passed=True, summary="green")]),
        handler=handler,
        on_event=events.append,
    )

    await orch.run()

    task_ends = [e for e in events if e["type"] == "task_end"]
    assert {e["task_id"] for e in task_ends} == {"ok", "bad"}
    for event in task_ends:
        assert isinstance(event["task_id"], str)
        assert isinstance(event["status"], str)
        assert not isinstance(event["status"], Enum)
        assert event["status"] in {"completed", "failed", "skipped"}
        json.dumps(event)
    statuses = {e["task_id"]: e["status"] for e in task_ends}
    assert statuses == {"ok": "completed", "bad": "failed"}


async def test_the_test_event_carries_the_full_verdict(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """``test`` is the event the UI renders as PASSED/FAILED from, so every field matters."""
    events: list[Event] = []
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(max_cycles=1),
        planner=CountingPlanner(),
        tester=FakeTester(
            [
                make_verdict(
                    passed=False,
                    summary="2 failed, 1 passed",
                    failures=["test_a FAILED", "test_b FAILED"],
                )
            ]
        ),
        handler=lambda worker: worker.written_files.append(f"{worker.task.id}.py"),
        on_event=events.append,
    )

    await orch.run()

    test_events = [e for e in events if e["type"] == "test"]
    assert len(test_events) == 1
    event = test_events[0]
    for key in ("passed", "skipped", "summary", "failures"):
        assert key in event
    assert event["passed"] is False
    assert event["skipped"] is False
    assert event["summary"] == "2 failed, 1 passed"
    assert event["failures"] == ["test_a FAILED", "test_b FAILED"]
    assert event["cycle"] == 1
    json.dumps(event)


def test_verdict_defaults_are_what_the_stop_conditions_read() -> None:
    """Guard: the loop's stop conditions read ``TestResult``'s defaults."""
    result = Verdict()
    assert result.passed is False
    assert result.skipped is False


# --------------------------------------------------------------------------- #
# The console mirror, driven by a real run
# --------------------------------------------------------------------------- #


def _rendered_run(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator_settings: Settings,
    buffer: io.StringIO,
) -> None:
    """Run one green cycle with a real rich console attached to the emitter."""
    orch = build_orchestrator(
        monkeypatch,
        orchestrator_settings,
        planner=FakePlanner([make_plan(make_task("t1", "write a.py"))]),
        tester=FakeTester([make_verdict(passed=True, summary="all green")]),
        handler=lambda worker: worker.written_files.append("a.py"),
    )
    console = Console(file=buffer, width=200)
    orch.console = console
    orch.emitter = EventEmitter(None, console=console)
    asyncio.run(orch.run())


def test_a_real_run_mirrors_every_event_type_to_the_console(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """End to end: a live run renders a line per event, and the types are wire values.

    This is the console half of the ``str(EventType)`` regression -- if the type field
    ever regresses to ``"EventType.LOG"`` the dispatch falls through and the whole run
    prints headers with no bodies.
    """
    buffer = io.StringIO()
    _rendered_run(monkeypatch, settings_factory(), buffer)
    out = buffer.getvalue()

    for etype in ("run_start", "cycle_start", "plan", "task_start", "task_end", "test"):
        assert f"{etype}: " in out, f"{etype} never rendered a line"
    assert "write a.py" in out
    assert "all green" in out
    assert "EventType." not in out
    assert "Traceback" not in out


def test_markup_in_llm_prose_cannot_break_the_console_mirror(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Callable[..., Settings]
) -> None:
    """Rich markup in planner/tester prose is escaped, so the render can never raise.

    ``_render_event`` feeds every dynamic segment through :func:`rich.markup.escape`
    precisely because an LLM will eventually emit ``[bold]``; without the escape that
    would raise ``MarkupError`` from inside ``console.print`` -- and, before the emitter
    quarantined subscriber failures, kill the run.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, width=200)
    orch = build_orchestrator(
        monkeypatch,
        settings_factory(),
        planner=FakePlanner(
            [make_plan(make_task("t1", "[red]write a.py[/red]"), rationale="[bold]because[/]")]
        ),
        tester=FakeTester([make_verdict(passed=True, summary="[bold]all green[/]")]),
        handler=lambda worker: worker.written_files.append("a.py"),
    )
    orch.console = console
    orch.emitter = EventEmitter(None, console=console)

    summary = asyncio.run(orch.run())

    assert summary.completed == 1
    out = buffer.getvalue()
    assert "[red]write a.py[/red]" in out
    assert "[bold]all green[/]" in out
    assert "EventType." not in out
    assert "MarkupError" not in out
