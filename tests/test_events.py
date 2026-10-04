"""Unit tests for `furrow.events`: the Event payload and the EventBus pub/sub contract."""

from __future__ import annotations

import asyncio
import json
from asyncio import Task
from collections.abc import AsyncGenerator, Awaitable
from typing import cast

import pytest

from furrow.events import Event, EventBus, EventType


def types_of(events: list[Event]) -> list[EventType]:
    return [event.type for event in events]


def messages_of(queue: asyncio.Queue[Event]) -> list[str]:
    return [queue.get_nowait().message for _ in range(queue.qsize())]


def open_stream(bus: EventBus) -> AsyncGenerator[Event, None]:
    """`stream()` is an async generator whose annotation hides `aclose`; restore it."""
    return cast("AsyncGenerator[Event, None]", bus.stream())


def open_stream_with(bus: EventBus, queue: asyncio.Queue[Event]) -> AsyncGenerator[Event, None]:
    return cast("AsyncGenerator[Event, None]", bus.stream(queue))


async def next_event(bus: EventBus, stream: AsyncGenerator[Event, None]) -> Task[Event]:
    """Start `stream` and wait until it has subscribed, then return the pending read.

    `stream()` is an async generator, so it subscribes lazily on its first iteration: an
    event emitted before then is missed, which is why the subscription is awaited here
    rather than assumed.
    """
    pending = asyncio.ensure_future(stream.__anext__())
    for _ in range(50):
        if bus._subscribers:
            return pending
        await asyncio.sleep(0)
    pending.cancel()
    raise AssertionError("stream() never subscribed to the bus")


async def wait_for(awaitable: Awaitable[Event]) -> Event:
    """Bound every stream read so a lost event fails the test instead of hanging it."""
    return await asyncio.wait_for(awaitable, timeout=5.0)


def test_event_type_members_are_distinct_strings() -> None:
    values = [member.value for member in EventType]
    assert len(values) == len(set(values)) == 11


def test_event_type_values() -> None:
    assert [member.value for member in EventType] == [
        "run_start",
        "run_end",
        "cycle_start",
        "cycle_end",
        "plan",
        "task_start",
        "task_end",
        "tool_call",
        "test_result",
        "log",
        "error",
    ]


def test_event_type_is_a_string_enum() -> None:
    assert EventType.ERROR == "error"
    assert isinstance(EventType.ERROR, str)


def test_event_defaults() -> None:
    event = Event(type=EventType.LOG)
    assert event.message == ""
    assert event.data == {}
    assert event.cycle == 0
    assert event.timestamp > 0.0


def test_event_data_is_not_shared_between_instances() -> None:
    first = Event(type=EventType.LOG)
    second = Event(type=EventType.LOG)
    first.data["k"] = "v"
    assert second.data == {}


def test_event_is_slotted() -> None:
    # slots=True keeps events small; a typo must fail loudly, not create a field.
    with pytest.raises(AttributeError):
        Event(type=EventType.LOG).unexpected = 1  # type: ignore[attr-defined]


def test_event_to_dict_shape() -> None:
    event = Event(type=EventType.TASK_END, message="done", cycle=3, data={"id": "1"})
    payload = event.to_dict()
    assert set(payload) == {"type", "message", "data", "cycle", "timestamp"}
    assert payload["type"] == "task_end"
    assert payload["message"] == "done"
    assert payload["data"] == {"id": "1"}
    assert payload["cycle"] == 3
    assert payload["timestamp"] == event.timestamp


def test_event_to_dict_is_json_serialisable() -> None:
    event = Event(type=EventType.PLAN, message="2 tasks", cycle=1, data={"rationale": "r"})
    assert json.loads(json.dumps(event.to_dict()))["data"] == {"rationale": "r"}


def test_event_to_dict_payload_matches_the_event() -> None:
    event = Event(type=EventType.TASK_START, message="task 1", data={"id": "1"})
    payload = event.to_dict()
    assert payload["data"] == event.data == {"id": "1"}
    payload["cycle"] = 99
    assert event.cycle == 0


def test_event_to_dict_copies_the_payload() -> None:
    # POLISH (wave 5b): "data" must not alias `event.data`, or a consumer mutating the
    # frame (the web layer serialises it straight to a socket) rewrites the stored event.
    event = Event(type=EventType.TASK_END, message="done", data={"id": "1", "status": "ok"})
    payload = event.to_dict()
    assert payload["data"] is not event.data
    payload["data"]["id"] = "tampered"
    payload["data"]["extra"] = True
    assert event.data == {"id": "1", "status": "ok"}
    # The copy is shallow by design: nested values are still shared, not deep-copied.
    nested = Event(type=EventType.LOG, data={"items": [1]})
    assert nested.to_dict()["data"]["items"] is nested.data["items"]


def test_emit_returns_the_recorded_event() -> None:
    bus = EventBus()
    emitted = bus.emit(EventType.LOG, "hello", cycle=2, note="n")
    assert emitted.type is EventType.LOG
    assert emitted.message == "hello"
    assert emitted.cycle == 2
    assert emitted.data == {"note": "n"}
    assert bus.history() == [emitted]


def test_emit_fans_out_to_every_subscriber() -> None:
    bus = EventBus()
    first = bus.subscribe()
    second = bus.subscribe()
    bus.emit(EventType.CYCLE_START, "cycle 1", cycle=1)
    assert first.get_nowait().type is EventType.CYCLE_START
    assert second.get_nowait().type is EventType.CYCLE_START


def test_emit_records_history_even_without_subscribers() -> None:
    bus = EventBus()
    bus.emit(EventType.ERROR, "boom")
    assert types_of(bus.history()) == [EventType.ERROR]


def test_history_is_oldest_first() -> None:
    bus = EventBus()
    for index in range(5):
        bus.emit(EventType.LOG, str(index))
    assert [event.message for event in bus.history()] == ["0", "1", "2", "3", "4"]


def test_history_evicts_the_oldest_beyond_maxlen() -> None:
    bus = EventBus(history=3)
    for index in range(5):
        bus.emit(EventType.LOG, str(index))
    assert [event.message for event in bus.history()] == ["2", "3", "4"]


def test_history_maxlen_zero_clamps_to_one() -> None:
    bus = EventBus(history=0)
    bus.emit(EventType.LOG, "first")
    bus.emit(EventType.LOG, "second")
    assert [event.message for event in bus.history()] == ["second"]


def test_history_negative_clamps_to_one() -> None:
    bus = EventBus(history=-5)
    bus.emit(EventType.LOG, "first")
    bus.emit(EventType.LOG, "second")
    assert len(bus.history()) == 1


def test_subscribe_does_not_replay_history() -> None:
    bus = EventBus()
    bus.emit(EventType.LOG, "before")
    queue = bus.subscribe()
    assert queue.empty()


def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    bus.unsubscribe(queue)
    bus.emit(EventType.LOG, "after")
    assert queue.empty()
    assert len(bus.history()) == 1


def test_unsubscribe_is_idempotent() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    bus.unsubscribe(queue)
    bus.unsubscribe(queue)
    assert bus._subscribers == []


def test_unsubscribe_of_an_unknown_queue_is_a_no_op() -> None:
    bus = EventBus()
    orphan = asyncio.Queue[Event]()
    bus.unsubscribe(orphan)
    assert bus._subscribers == []


def test_emit_survives_a_full_subscriber_queue() -> None:
    bus = EventBus()
    full = bus.subscribe()
    healthy = bus.subscribe()
    for index in range(full.maxsize):
        full.put_nowait(Event(type=EventType.LOG, message=str(index)))
    assert full.full()
    bus.emit(EventType.ERROR, "still delivered")  # must not raise
    assert healthy.get_nowait().message == "still delivered"
    assert types_of(bus.history()) == [EventType.ERROR]


def test_clear_empties_history() -> None:
    bus = EventBus()
    bus.emit(EventType.LOG, "a")
    bus.clear()
    assert bus.history() == []


def test_clear_leaves_subscribers_attached() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    bus.emit(EventType.LOG, "a")
    bus.clear()
    assert messages_of(queue) == ["a"]  # live delivery is not rewound by clear()
    bus.emit(EventType.LOG, "b")
    assert messages_of(queue) == ["b"]


def test_subscriber_queues_are_bounded() -> None:
    bus = EventBus()
    assert bus.subscribe().maxsize == 1000


async def test_stream_yields_events_in_order() -> None:
    bus = EventBus()
    stream = open_stream(bus)
    first = await next_event(bus, stream)
    bus.emit(EventType.PLAN, "planned 1 task(s)")
    bus.emit(EventType.LOG, "log")
    assert (await wait_for(first)).message == "planned 1 task(s)"
    assert (await wait_for(stream.__anext__())).message == "log"
    await stream.aclose()


async def test_stream_auto_unsubscribes_on_aclose() -> None:
    bus = EventBus()
    stream = open_stream(bus)
    first = await next_event(bus, stream)
    bus.emit(EventType.LOG, "one")
    assert (await wait_for(first)).message == "one"
    await stream.aclose()
    assert bus._subscribers == []


async def test_stream_keeps_a_caller_supplied_queue_subscribed() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    stream = open_stream_with(bus, queue)
    bus.emit(EventType.LOG, "one")
    assert (await wait_for(stream.__anext__())).message == "one"
    await stream.aclose()
    assert bus._subscribers == [queue]
    bus.emit(EventType.LOG, "two")
    assert queue.get_nowait().message == "two"
