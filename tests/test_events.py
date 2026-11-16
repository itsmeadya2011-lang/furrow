"""Tests for the event bus."""

import json
import time

import pytest

from furrow.core.events import MAX_QUEUE_SIZE, Event, EventBus, EventType


def test_event_creates_timestamp_and_serializes_type() -> None:
    event = Event(type=EventType.LOG, message="hi")
    assert event.timestamp
    assert isinstance(event.timestamp, str)
    result = event.to_dict()
    assert result["type"] == "log"


def test_to_dict_has_required_keys_and_is_json_serialisable() -> None:
    event = Event(type=EventType.TASK_STARTED, message="start", data={"task_id": 1})
    result = event.to_dict()
    assert "message" in result
    assert "data" in result
    assert "timestamp" in result
    json.dumps(result)


def test_subscribe_returns_queue_and_subscribers_are_independent() -> None:
    bus = EventBus()
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    assert q1 is not q2


@pytest.mark.asyncio
async def test_emit_puts_event_on_all_subscribers() -> None:
    bus = EventBus()
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    event = Event(type=EventType.LOG, message="hello")
    bus.emit(event)
    got1 = await q1.get()
    got2 = await q2.get()
    assert got1 is event
    assert got2 is event


def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    q1 = bus.subscribe()
    event = Event(type=EventType.LOG, message="x")
    bus.unsubscribe(q1)
    bus.emit(event)
    assert q1.empty()


def test_unsubscribe_unknown_queue_does_not_raise() -> None:
    bus = EventBus()
    q = __import__("asyncio").Queue()
    bus.unsubscribe(q)


def test_emit_with_no_subscribers_does_not_raise() -> None:
    bus = EventBus()
    bus.emit(Event(type=EventType.LOG, message="noop"))


def test_clear_empties_subscribers() -> None:
    bus = EventBus()
    bus.subscribe()
    bus.subscribe()
    assert bus.subscriber_count == 2
    bus.clear()
    assert bus.subscriber_count == 0


def test_eventtype_cycle_started_value() -> None:
    assert EventType.CYCLE_STARTED.value == "cycle_started"


def test_subscriber_queue_is_bounded() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    assert queue.maxsize > 0
    assert queue.maxsize == MAX_QUEUE_SIZE


def test_emit_overflow_does_not_raise_and_queue_stays_bounded() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    start = time.monotonic()
    for i in range(MAX_QUEUE_SIZE * 5):
        bus.emit(Event(type=EventType.LOG, message=str(i)))
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, "emit must not block on a full queue"
    assert queue.qsize() <= queue.maxsize


def test_newest_event_survives_overfill() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    events = [Event(type=EventType.LOG, message=str(i)) for i in range(MAX_QUEUE_SIZE + 10)]
    for event in events:
        bus.emit(event)
    drained: list[Event] = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert len(drained) == MAX_QUEUE_SIZE
    assert drained[-1] is events[-1]
    assert all(drained_event is not events[0] for drained_event in drained)


def test_dropped_count_is_zero_in_normal_case_and_counts_overflow_drops() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    for i in range(10):
        bus.emit(Event(type=EventType.LOG, message=str(i)))
    assert bus.dropped_count == 0
    assert bus.dropped_for(queue) == 0
    for _i in range(MAX_QUEUE_SIZE + 5):
        bus.emit(Event(type=EventType.LOG, message="flood"))
    # 10 + (MAX_QUEUE_SIZE + 5) events were emitted into a MAX_QUEUE_SIZE queue.
    assert bus.dropped_count == 15
    assert bus.dropped_for(queue) == 15


@pytest.mark.asyncio
async def test_emit_async_delivers_event_to_subscribers() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    event = Event(type=EventType.LOG, message="async")
    await bus.emit_async(event)
    got = await queue.get()
    assert got is event


def test_emit_counts_every_drop_even_when_put_fails_again() -> None:
    """Every event that cannot be queued is counted as a drop, including the
    retry after the oldest event was evicted."""
    import asyncio as _asyncio

    bus = EventBus()
    queue = _asyncio.Queue(maxsize=1)
    bus._subscribers.append(queue)

    # Fill the queue so the first put_nowait raises QueueFull.
    queue.put_nowait(Event(type=EventType.LOG, message="first"))
    assert queue.full()

    bus.emit(Event(type=EventType.LOG, message="second"))
    bus.emit(Event(type=EventType.LOG, message="third"))

    # The first emit dropped the oldest ("first") and queued "second".
    # The second emit dropped "second" and queued "third".
    assert bus.dropped_count == 2
    assert bus.dropped_for(queue) == 2
    assert queue.qsize() == 1
    drained = queue.get_nowait()
    assert drained.message == "third"


def test_unsubscribe_removes_every_occurrence_of_the_queue() -> None:
    """A queue registered twice is fully removed by a single unsubscribe."""
    import asyncio as _asyncio

    bus = EventBus()
    queue = _asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    # Register the same queue object twice.
    bus._subscribers.append(queue)
    bus._subscribers.append(queue)
    assert bus.subscriber_count == 2

    bus.unsubscribe(queue)
    assert bus.subscriber_count == 0

    # A second unsubscribe is still a no-op.
    bus.unsubscribe(queue)
    assert bus.subscriber_count == 0
