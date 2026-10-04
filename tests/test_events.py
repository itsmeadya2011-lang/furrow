"""Behavioural tests for :mod:`furrow.core.events`.

The event bus is the seam every other layer is observed through, so these tests
pin down the three promises its docstrings make and that the rest of Furrow
relies on:

* **Fan-out.** One event reaches every subscriber -- sync handlers and coroutine
  handlers alike -- via ``await bus.emit(...)``.
* **Isolation.** A subscriber that raises is logged with
  ``logging.exception`` and swallowed. It cannot stop the handlers behind it and
  cannot propagate into the loop, because "one bad subscriber must not break the
  run" is the whole point of a fan-out bus.
* **Mutability during dispatch.** ``subscribe``/``unsubscribe`` are legal while
  an event is being delivered, so dispatch iterates a snapshot of the handler
  list. That is what makes the ``EventCollector`` teardown path safe.

``emit_nowait`` is the sync-only variant. Its contract is narrower on purpose:
it skips coroutine handlers and *warns* rather than running them, and a sync
handler that returns an awaitable has that awaitable closed. Leaking an
unawaited coroutine is the failure mode worth pinning here, so it is detected
with ``warnings.catch_warnings`` + ``gc.collect()`` -- plus a positive control
test proving that detector can actually see such a warning.

The factory functions are covered in two directions: every one of the twelve
builds the ``EventType`` its name promises, and ``plan_created`` carries the
cycle that produced it (it used to ignore it, so every plan event was filed
under cycle 0 and any consumer grouping the stream by cycle mis-attributed it).

Everything here is offline and deterministic: no network, no API key.
"""

from __future__ import annotations

import gc
import json
import logging
import warnings
from collections.abc import Callable
from typing import Any

import pytest

from furrow.core.events import (
    Event,
    EventBus,
    EventType,
    cycle_completed,
    cycle_started,
    log,
    plan_created,
    run_completed,
    run_failed,
    run_started,
    task_completed,
    task_failed,
    task_skipped,
    task_started,
    tests_completed,
)
from tests.conftest import EventCollector

EVENTS_LOGGER = "furrow.core.events"

# The ``tests_completed`` factory matches pytest's default ``test*`` collection
# pattern, so pytest would collect it as a test function and then fail on its
# ``passed``/``summary``/``failures`` parameters (which are not fixtures).
# Marking it non-collectable silences that -- pytest's PyCollector.isnosetest()
# checks exactly this attribute. The flag is set on the imported function
# object, not in furrow/core/events.py; conftest.py does the same thing for
# ``furrow.config.TestResult``.
tests_completed.__test__ = False

#: A plan payload nested two levels deep, so the JSON checks exercise real
#: containers rather than a flat string dict.
PLAN: dict[str, Any] = {
    "tasks": [{"id": "1", "description": "write it", "files": ["a.py"]}],
    "rationale": "the goal needs it",
}


def _never_awaited_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    """The ``coroutine ... was never awaited`` RuntimeWarnings in ``caught``."""
    return [
        str(entry.message)
        for entry in caught
        if issubclass(entry.category, RuntimeWarning) and "never awaited" in str(entry.message)
    ]


# --- fan-out: sync and async handlers both run ---


async def test_emit_delivers_to_sync_and_async_handlers(
    bus: EventBus, collector: EventCollector
) -> None:
    seen: list[tuple[str, Event]] = []

    def sync_handler(event: Event) -> None:
        seen.append(("sync", event))

    async def async_handler(event: Event) -> None:
        seen.append(("async", event))

    bus.subscribe(sync_handler)
    bus.subscribe(async_handler)

    event = cycle_started(3)
    await bus.emit(event)

    # Both ran, in subscription order, and each was handed the emitted object
    # itself (not a copy) -- the sync path awaits nothing, the async path does.
    assert [name for name, _ in seen] == ["sync", "async"]
    assert all(received is event for _, received in seen)
    # The conftest collector is a third (sync) subscriber on the same bus.
    assert collector.events == [event]


async def test_emit_awaits_a_sync_handler_that_returns_an_awaitable(bus: EventBus) -> None:
    # A plain ``async def`` handler is not the only awaitable shape: a sync
    # callable (or a functools.partial wrapping one) that *returns* a coroutine
    # must still have it awaited.
    completed: list[str] = []

    def returns_coroutine(event: Event) -> Any:
        async def inner() -> None:
            completed.append(event.type.value)

        return inner()

    bus.subscribe(returns_coroutine)
    await bus.emit(task_completed("1", "done", cycle=1))

    assert completed == ["task.completed"]


async def test_a_raising_handler_is_logged_and_the_rest_still_run(
    bus: EventBus, caplog: pytest.LogCaptureFixture
) -> None:
    ran: list[str] = []

    def boom(event: Event) -> None:
        raise RuntimeError("subscriber exploded")

    def after_sync(event: Event) -> None:
        ran.append("sync-after")

    async def after_async(event: Event) -> None:
        ran.append("async-after")

    bus.subscribe(boom)
    bus.subscribe(after_sync)
    bus.subscribe(after_async)

    emitted = task_failed("3", "nope", 1.5, cycle=2)
    with caplog.at_level(logging.ERROR, logger=EVENTS_LOGGER):
        # emit() swallows handler exceptions by contract, so this must return
        # normally rather than raising out of the loop.
        await bus.emit(emitted)

    # The failing handler did not stop the two behind it.
    assert ran == ["sync-after", "async-after"]

    failures = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(failures) == 1
    record = failures[0]
    assert record.name == EVENTS_LOGGER
    assert "Event handler failed" in record.getMessage()
    # The event type is logged lazily as the %s arg. ``str()`` of a ``str,
    # Enum`` member is version-dependent ("EventType.TASK_FAILED" on 3.10, the
    # bare value on newer interpreters), so assert against str() itself, and
    # pin the arg identity.
    assert str(EventType.TASK_FAILED) in record.getMessage()
    assert record.args == (emitted.type,)
    # logging.exception attaches the traceback, so the cause stays diagnosable.
    assert record.exc_info is not None
    assert "subscriber exploded" in caplog.text


# --- subscribe / unsubscribe ---


async def test_subscribe_returns_a_callable_that_unsubscribes(bus: EventBus) -> None:
    received: list[Event] = []

    def handler(event: Event) -> None:
        received.append(event)

    unsubscribe = bus.subscribe(handler)
    assert callable(unsubscribe)
    assert bus.handler_count == 1

    await bus.emit(log("first"))
    assert len(received) == 1

    unsubscribe()
    assert bus.handler_count == 0

    await bus.emit(log("second"))
    assert len(received) == 1, "an unsubscribed handler still received the event"

    # Idempotent: calling it again is a no-op, not a ValueError.
    unsubscribe()
    assert bus.handler_count == 0


async def test_subscribe_and_unsubscribe_during_dispatch_use_a_snapshot(bus: EventBus) -> None:
    order: list[str] = []

    def late(event: Event) -> None:
        order.append("late")

    def departing(event: Event) -> None:
        order.append("departing")

    def churner(event: Event) -> None:
        order.append("churner")
        bus.subscribe(late)  # added mid-dispatch
        bus.unsubscribe(departing)  # removed mid-dispatch

    bus.subscribe(churner)
    bus.subscribe(departing)

    await bus.emit(log("first"))

    # Dispatch iterated a snapshot taken before the loop: `departing` was in it
    # so it still ran (removal is deferred), while `late` was not in it so it
    # did NOT run this round. Neither mutation raised.
    assert order == ["churner", "departing"]
    assert bus.handler_count == 2

    order.clear()
    await bus.emit(log("second"))

    # The second dispatch sees the mutated list: churner plus late, not
    # departing.
    assert order == ["churner", "late"]


async def test_handler_count_tracks_subscribe_and_unsubscribe(bus: EventBus) -> None:
    assert bus.handler_count == 0

    def first(event: Event) -> None:
        return None

    def second(event: Event) -> None:
        return None

    # Removing a handler that was never registered is safe, even on an empty
    # bus: unsubscribe() suppresses the ValueError list.remove would raise.
    bus.unsubscribe(first)
    bus.unsubscribe(second)
    bus.unsubscribe(lambda event: None)
    assert bus.handler_count == 0

    bus.subscribe(first)
    bus.subscribe(second)
    assert bus.handler_count == 2

    bus.unsubscribe(first)
    assert bus.handler_count == 1

    # Already-removed is equally safe.
    bus.unsubscribe(first)
    assert bus.handler_count == 1

    bus.unsubscribe(second)
    assert bus.handler_count == 0


# --- emit_nowait: the sync-only path ---


def test_emit_nowait_delivers_to_sync_handlers_and_skips_async_ones(
    bus: EventBus, collector: EventCollector, caplog: pytest.LogCaptureFixture
) -> None:
    ran: list[str] = []

    def sync_handler(event: Event) -> None:
        ran.append(f"sync:{event.type.value}")

    async def async_handler(event: Event) -> None:
        ran.append(f"async:{event.type.value}")

    bus.subscribe(sync_handler)
    bus.subscribe(async_handler)

    with caplog.at_level(logging.WARNING, logger=EVENTS_LOGGER):
        bus.emit_nowait(run_started("go"))

    assert ran == ["sync:run.started"]
    assert collector.types() == ["run.started"]
    # The skipped coroutine handler is reported, not silently dropped.
    assert "Skipping async handler" in caplog.text


def test_emit_nowait_leaves_no_unawaited_coroutine(bus: EventBus) -> None:
    ran: list[str] = []

    async def async_handler(event: Event) -> None:
        ran.append("async")

    bus.subscribe(async_handler)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bus.emit_nowait(log("sync-only"))
        gc.collect()

    # The handler was never even called, so there is no coroutine object left
    # for the interpreter to complain about at collection time.
    assert ran == []
    assert _never_awaited_warnings(caught) == []


def test_emit_nowait_closes_an_awaitable_a_sync_handler_returned(
    bus: EventBus, caplog: pytest.LogCaptureFixture
) -> None:
    ran: list[str] = []

    def returns_coroutine(event: Event) -> Any:
        async def inner() -> None:
            ran.append("inner")

        return inner()

    bus.subscribe(returns_coroutine)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.WARNING, logger=EVENTS_LOGGER):
            bus.emit_nowait(log("sync-only"))
        gc.collect()

    # A sync caller cannot await it, so it must be closed rather than dropped --
    # either way no body runs and no "never awaited" warning escapes.
    assert ran == []
    assert _never_awaited_warnings(caught) == []
    assert "cannot await it" in caplog.text


def test_the_unawaited_coroutine_detector_can_see_a_real_leak() -> None:
    # Positive control for the two tests above: if the capture were wrong they
    # would pass vacuously. Dropping a coroutine on the floor does register.
    async def never_awaited() -> None:
        """Docstring-only body: the coroutine below is deliberately dropped."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pending = never_awaited()
        del pending
        gc.collect()

    assert _never_awaited_warnings(caught) != []


def test_emit_nowait_swallows_and_logs_a_raising_handler(
    bus: EventBus, caplog: pytest.LogCaptureFixture
) -> None:
    ran: list[str] = []

    def boom(event: Event) -> None:
        raise RuntimeError("sync subscriber exploded")

    def after(event: Event) -> None:
        ran.append("after")

    bus.subscribe(boom)
    bus.subscribe(after)

    with caplog.at_level(logging.ERROR, logger=EVENTS_LOGGER):
        bus.emit_nowait(log("x"))

    assert ran == ["after"]
    assert "Sync event handler failed" in caplog.text
    assert "sync subscriber exploded" in caplog.text


# --- Event / to_dict ---


def test_event_type_covers_the_twelve_loop_events() -> None:
    assert {member.name: member.value for member in EventType} == {
        "RUN_STARTED": "run.started",
        "RUN_COMPLETED": "run.completed",
        "RUN_FAILED": "run.failed",
        "CYCLE_STARTED": "cycle.started",
        "PLAN_CREATED": "plan.created",
        "TASK_STARTED": "task.started",
        "TASK_COMPLETED": "task.completed",
        "TASK_FAILED": "task.failed",
        "TASK_SKIPPED": "task.skipped",
        "TESTS_COMPLETED": "tests.completed",
        "CYCLE_COMPLETED": "cycle.completed",
        "LOG": "log",
    }
    # EventType is a str enum, so consumers can compare against the raw value.
    assert EventType.LOG == "log"


def test_to_dict_is_json_serialisable_flat_and_nested() -> None:
    event = plan_created(PLAN, cycle=4)
    payload = event.to_dict()

    assert set(payload) == {"type", "cycle", "timestamp", "data"}
    assert payload["type"] == "plan.created"
    assert payload["cycle"] == 4
    assert payload["timestamp"] == event.timestamp
    assert payload["data"] == {"plan": PLAN}

    # json.dumps must accept it, nested containers included, and the decoded
    # payload must compare equal to the original.
    restored = json.loads(json.dumps(payload))
    assert restored == payload
    assert restored["type"] == EventType.PLAN_CREATED.value
    assert restored["data"]["plan"]["tasks"][0]["files"] == ["a.py"]

    # Defaults serialise too, so a bare event is safe to put on a stream.
    bare = Event(EventType.LOG)
    assert bare.data == {}
    assert bare.cycle == 0
    assert json.loads(json.dumps(bare.to_dict())) == {
        "type": "log",
        "cycle": 0,
        "timestamp": bare.timestamp,
        "data": {},
    }


def test_two_events_do_not_share_the_default_data_dict() -> None:
    # ``data`` is a per-instance default, so a mutation on one event must not
    # bleed into the next event built with no data at all.
    first = Event(EventType.LOG)
    second = Event(EventType.LOG)
    first.data["only"] = "mine"

    assert second.data == {}
    assert second.to_dict()["data"] == {}


# --- factory functions ---


#: A factory case: the factory, its positional args, its keyword args, the
#: ``EventType`` it must build, and the cycle it must stamp.
FactoryCase = tuple[Callable[..., Event], tuple[Any, ...], dict[str, Any], EventType, int]

#: ``run_*`` events are run-scoped and carry no cycle; every per-cycle factory
#: stamps the cycle it belongs to.
FACTORY_CASES: list[FactoryCase] = [
    (run_started, ("ship it",), {"max_cycles": 3}, EventType.RUN_STARTED, 0),
    (run_completed, (2, "goal reached"), {}, EventType.RUN_COMPLETED, 0),
    (run_failed, ("boom",), {}, EventType.RUN_FAILED, 0),
    (cycle_started, (1,), {}, EventType.CYCLE_STARTED, 1),
    (plan_created, (PLAN,), {"cycle": 9}, EventType.PLAN_CREATED, 9),
    (task_started, ("1", "do it", ["a.py"]), {"cycle": 2}, EventType.TASK_STARTED, 2),
    (task_completed, ("1", "ok", 1.25), {"cycle": 2}, EventType.TASK_COMPLETED, 2),
    (task_failed, ("1", "bad", 0.5), {"cycle": 2}, EventType.TASK_FAILED, 2),
    (task_skipped, ("1", "no diff"), {"cycle": 2}, EventType.TASK_SKIPPED, 2),
    (tests_completed, (True, "5 passed", []), {"cycle": 2}, EventType.TESTS_COMPLETED, 2),
    (cycle_completed, (2, True), {}, EventType.CYCLE_COMPLETED, 2),
    (log, ("hello",), {"level": "warning", "cycle": 2}, EventType.LOG, 2),
]


@pytest.mark.parametrize(
    ("factory", "args", "kwargs", "expected", "expected_cycle"),
    FACTORY_CASES,
    ids=[case[0].__name__ for case in FACTORY_CASES],
)
def test_factories_build_the_expected_event_type(
    factory: Callable[..., Event],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    expected: EventType,
    expected_cycle: int,
) -> None:
    event = factory(*args, **kwargs)

    assert isinstance(event, Event)
    assert event.type is expected
    assert event.cycle == expected_cycle
    assert event.data, "every factory populates data"
    # Whatever the payload, a factory-built event is streamable as-is.
    assert json.loads(json.dumps(event.to_dict()))["type"] == expected.value


async def test_plan_created_is_stamped_with_the_cycle_that_produced_it(
    bus: EventBus, collector: EventCollector
) -> None:
    # Regression guard. ``plan_created`` used to drop its cycle argument, so
    # *every* plan.created event was attributed to cycle 0. Any consumer that
    # groups the event stream by cycle (the web dashboard) then filed all
    # plans under the first cycle.
    assert plan_created(PLAN, cycle=7).cycle == 7
    assert plan_created(PLAN).cycle == 0
    # The plan payload itself is untouched by the cycle stamp.
    assert plan_created(PLAN, cycle=7).data == {"plan": PLAN}
    # ``cycle`` is keyword-only, so a positional call cannot silently "work".
    with pytest.raises(TypeError):
        plan_created(PLAN, 7)

    second_plan: dict[str, Any] = {"tasks": [], "rationale": "no further work"}
    await bus.emit(plan_created(PLAN, cycle=1))
    await bus.emit(plan_created(second_plan, cycle=2))

    plans = collector.of_type(EventType.PLAN_CREATED)
    assert [event.cycle for event in plans] == [1, 2], "plans must not all land in cycle 0"
    assert [event.data["plan"] for event in plans] == [PLAN, second_plan]
