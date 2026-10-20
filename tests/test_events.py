"""Tests for the event bus."""

import json

import pytest

from furrow.core.events import Event, EventBus, EventType


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
