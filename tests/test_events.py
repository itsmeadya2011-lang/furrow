"""Tests for the structured event bus.

Two consumers depend on this module's guarantees, and both fail in confusing ways if a
guarantee is quietly dropped: the browser needs every event to be JSON-serialisable with
a real event type (not a Python repr of one), and the orchestrator needs a broken
subscriber to be harmless. The tests below pin those behaviours rather than the
implementation shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from furrow.core.events import Event, EventEmitter, EventType, event_to_dict


class Sample(BaseModel):
    name: str
    count: int


@dataclass
class Point:
    x: int
    y: int


def test_emit_records_type_message_data_and_float_ts():
    emitter = EventEmitter()

    emitter.emit(EventType.CYCLE, "cycle 1", index=1, goal="ship it")

    events = emitter.events
    assert len(events) == 1
    event = events[0]
    assert event.type is EventType.CYCLE
    assert event.message == "cycle 1"
    assert event.data == {"index": 1, "goal": "ship it"}
    assert isinstance(event.ts, float)
    assert event.ts > 0


def test_emit_without_data_records_an_empty_dict():
    emitter = EventEmitter()

    emitter.emit(EventType.LOG, "plain message")

    assert emitter.events[0].data == {}


def test_data_default_is_not_shared_between_events():
    """A plain `{}` class default would be created once and handed to every event."""
    emitter = EventEmitter()

    emitter.emit(EventType.LOG, "first")
    emitter.events[0].data["mutated"] = True
    emitter.emit(EventType.LOG, "second")

    assert emitter.events[1].data == {}


def test_event_dataclass_defaults_are_independent_instances():
    first = Event(type=EventType.LOG, message="a")
    second = Event(type=EventType.LOG, message="b")

    first.data["x"] = 1

    assert second.data == {}
    assert first.ts != second.ts or isinstance(first.ts, float)


def test_subscriber_receives_events_in_emission_order():
    seen: list[Event] = []
    emitter = EventEmitter(seen.append)

    emitter.emit(EventType.CYCLE, "one")
    emitter.emit(EventType.TASK_START, "two")
    emitter.emit(EventType.DONE, "three")

    assert [event.message for event in seen] == ["one", "two", "three"]
    assert [event.type for event in seen] == [
        EventType.CYCLE,
        EventType.TASK_START,
        EventType.DONE,
    ]


def test_constructor_callback_and_on_both_deliver():
    first: list[Event] = []
    second: list[Event] = []
    emitter = EventEmitter(first.append)

    assert emitter.on(second.append) is None

    emitter.emit(EventType.PLAN, "planned")

    assert len(first) == 1
    assert len(second) == 1


def test_multiple_subscribers_all_receive_in_registration_order():
    order: list[str] = []
    emitter = EventEmitter(lambda event: order.append("first"))
    emitter.on(lambda event: order.append("second"))
    emitter.on(lambda event: order.append("third"))

    emitter.emit(EventType.TEST, "pytest", returncode=0)

    assert order == ["first", "second", "third"]


def test_raising_subscriber_does_not_break_emit_or_starve_later_subscribers():
    """This is why `emit` isolates callbacks: observability must not fail the work."""
    order: list[str] = []

    def boom(event: Event) -> None:
        order.append("boom")
        raise RuntimeError("websocket client vanished")

    emitter = EventEmitter(boom)
    emitter.on(lambda event: order.append("after"))

    emitter.emit(EventType.LOG, "still delivered")

    assert order == ["boom", "after"]
    assert len(emitter.events) == 1


def test_raising_subscriber_does_not_prevent_the_event_being_recorded():
    def boom(event: Event) -> None:
        raise ValueError("nope")

    emitter = EventEmitter(boom)

    emitter.emit(EventType.ERROR, "recorded anyway")

    assert emitter.events[0].message == "recorded anyway"


def test_subscribers_registered_during_emit_receive_the_next_event():
    later: list[Event] = []
    emitter = EventEmitter(lambda event: emitter.on(later.append))

    emitter.emit(EventType.LOG, "first")

    assert later == []
    emitter.emit(EventType.LOG, "second")
    assert [event.message for event in later] == ["second"]


def test_history_is_bounded_and_drops_the_oldest():
    emitter = EventEmitter(history_limit=3)

    for index in range(5):
        emitter.emit(EventType.LOG, f"m{index}")

    assert len(emitter.events) == 3
    assert [event.message for event in emitter.events] == ["m2", "m3", "m4"]


def test_default_history_limit_is_finite():
    emitter = EventEmitter()

    for _ in range(1500):
        emitter.emit(EventType.LOG, "spam")

    assert len(emitter.events) == 1000
    assert emitter.events[0].message == "spam"
    assert emitter.events[-1].message == "spam"


def test_history_limit_of_one_keeps_only_the_newest():
    emitter = EventEmitter(history_limit=1)

    emitter.emit(EventType.LOG, "old")
    emitter.emit(EventType.LOG, "new")

    assert [event.message for event in emitter.events] == ["new"]


@pytest.mark.parametrize("limit", [0, -1, -100])
def test_non_positive_history_limit_is_rejected(limit: int):
    with pytest.raises(ValueError):
        EventEmitter(history_limit=limit)


def test_clear_empties_history_but_keeps_subscribers():
    seen: list[Event] = []
    emitter = EventEmitter(seen.append)
    emitter.emit(EventType.LOG, "before")

    emitter.clear()

    assert emitter.events == []
    emitter.emit(EventType.LOG, "after")
    assert [event.message for event in seen] == ["before", "after"]
    assert len(emitter.events) == 1


def test_events_property_returns_a_snapshot_not_the_deque():
    emitter = EventEmitter(history_limit=2)
    emitter.emit(EventType.LOG, "a")

    snapshot = emitter.events
    snapshot.append(Event(type=EventType.LOG, message="injected"))

    assert len(emitter.events) == 1


def test_of_type_filters_and_preserves_order():
    emitter = EventEmitter()
    emitter.emit(EventType.CYCLE, "c")
    emitter.emit(EventType.TASK_DONE, "t1")
    emitter.emit(EventType.LOG, "l")
    emitter.emit(EventType.TASK_DONE, "t2")
    emitter.emit(EventType.ERROR, "e")

    assert [event.message for event in emitter.of_type(EventType.TASK_DONE)] == ["t1", "t2"]
    assert [
        event.message
        for event in emitter.of_type(EventType.CYCLE, EventType.ERROR, EventType.LOG)
    ] == ["c", "l", "e"]
    assert len(emitter.of_type(EventType.CYCLE, EventType.ERROR, EventType.LOG)) == 3


def test_of_type_returns_empty_for_a_type_never_emitted():
    emitter = EventEmitter()
    emitter.emit(EventType.LOG, "only")

    assert emitter.of_type(EventType.TEST) == []
    assert emitter.of_type() == []


def test_of_type_respects_the_history_bound():
    emitter = EventEmitter(history_limit=2)
    emitter.emit(EventType.LOG, "dropped")
    emitter.emit(EventType.PLAN, "kept")
    emitter.emit(EventType.PLAN, "kept too")

    assert [event.message for event in emitter.of_type(EventType.PLAN)] == ["kept", "kept too"]
    assert emitter.of_type(EventType.LOG) == []


def test_event_to_dict_is_json_serialisable():
    emitter = EventEmitter()
    emitter.emit(EventType.TASK_DONE, "worker finished", files=["a.py"], count=3)

    payload = event_to_dict(emitter.events[0])

    assert set(payload) == {"type", "message", "data", "ts"}
    assert json.loads(json.dumps(payload)) == payload


def test_event_type_serialises_as_its_value_not_its_repr():
    """`str(EventType.LOG)` is "EventType.LOG" on 3.10; the wire needs "log"."""
    emitter = EventEmitter()

    emitter.emit(EventType.LOG, "hi")
    payload = event_to_dict(emitter.events[0])

    assert payload["type"] == "log"
    assert payload["type"] != "EventType.LOG"
    assert json.dumps(payload["type"]) == '"log"'


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [
        (EventType.CYCLE, "cycle"),
        (EventType.PLAN, "plan"),
        (EventType.TASK_START, "task_start"),
        (EventType.TASK_DONE, "task_done"),
        (EventType.TEST, "test"),
        (EventType.LOG, "log"),
        (EventType.ERROR, "error"),
        (EventType.DONE, "done"),
    ],
)
def test_every_event_type_serialises_to_its_wire_token(
    event_type: EventType, expected: str
):
    assert event_to_dict(Event(type=event_type, message="m"))["type"] == expected


def test_event_to_dict_handles_an_empty_data_payload():
    payload = event_to_dict(Event(type=EventType.DONE, message="finished"))

    assert payload == {
        "type": "done",
        "message": "finished",
        "data": {},
        "ts": payload["ts"],
    }
    assert isinstance(payload["ts"], float)


def test_nested_containers_are_converted_recursively():
    event = Event(
        type=EventType.TEST,
        message="nested",
        data={"outer": {"inner": [1, ("two", 2.5), {"three": True}]}},
    )

    data = event_to_dict(event)["data"]

    assert data == {"outer": {"inner": [1, ["two", 2.5], {"three": True}]}}
    json.dumps(data)


def test_set_of_strings_is_sorted_for_deterministic_output():
    event = Event(type=EventType.TASK_DONE, message="files", data={"paths": {"b.py", "a.py"}})

    assert event_to_dict(event)["data"]["paths"] == ["a.py", "b.py"]


def test_frozenset_of_strings_is_sorted_too():
    event = Event(type=EventType.TASK_DONE, message="files", data={"paths": frozenset({"b", "a"})})

    assert event_to_dict(event)["data"]["paths"] == ["a", "b"]


def test_set_of_non_strings_keeps_list_order_and_contents():
    event = Event(type=EventType.TEST, message="codes", data={"codes": {7, 3}})

    converted = event_to_dict(event)["data"]["codes"]

    assert isinstance(converted, list)
    assert sorted(converted) == [3, 7]
    json.dumps(converted)


def test_set_of_enums_converts_to_sorted_plain_strings():
    event = Event(
        type=EventType.LOG,
        message="types",
        data={"types": {EventType.LOG, EventType.ERROR}},
    )

    assert event_to_dict(event)["data"]["types"] == ["error", "log"]


def test_paths_become_strings():
    event = Event(type=EventType.TASK_DONE, message="wrote", data={"path": Path("/tmp/x/y.py")})

    assert event_to_dict(event)["data"]["path"] == "/tmp/x/y.py"


def test_datetimes_and_dates_become_iso_strings():
    stamp = datetime(2026, 10, 5, 2, 50, 34, tzinfo=timezone.utc)
    event = Event(
        type=EventType.CYCLE,
        message="stamped",
        data={"at": stamp, "day": date(2026, 10, 5)},
    )

    data = event_to_dict(event)["data"]

    assert data["at"] == "2026-10-05T02:50:34+00:00"
    assert data["day"] == "2026-10-05"


def test_bytes_are_decoded_with_replacement_not_raised():
    event = Event(type=EventType.TEST, message="output", data={"raw": b"passed\xff"})

    converted = event_to_dict(event)["data"]["raw"]

    assert isinstance(converted, str)
    assert converted.startswith("passed")
    json.dumps(converted)


def test_scalars_pass_through_unchanged():
    event = Event(
        type=EventType.LOG,
        message="scalars",
        data={"s": "text", "i": 42, "f": 1.5, "b": False, "none": None},
    )

    data = event_to_dict(event)["data"]

    assert data == {"s": "text", "i": 42, "f": 1.5, "b": False, "none": None}


def test_arbitrary_object_falls_back_to_repr():
    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque instance>"

    event = Event(type=EventType.ERROR, message="weird", data={"obj": Opaque()})

    assert event_to_dict(event)["data"]["obj"] == "<Opaque instance>"


def test_enum_from_an_unrelated_enum_also_yields_its_value():
    class Colour(str, Enum):
        RED = "red"

    event = Event(type=EventType.LOG, message="colour", data={"c": Colour.RED})

    assert event_to_dict(event)["data"]["c"] == "red"


def test_dataclasses_are_expanded():
    event = Event(type=EventType.PLAN, message="plan", data={"point": Point(1, 2)})

    assert event_to_dict(event)["data"]["point"] == {"x": 1, "y": 2}


def test_nested_dataclass_inside_a_list_is_expanded():
    event = Event(type=EventType.PLAN, message="plan", data={"points": [Point(1, 2), Point(3, 4)]})

    assert event_to_dict(event)["data"]["points"] == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]


def test_pydantic_model_is_dumped():
    event = Event(
        type=EventType.TASK_DONE,
        message="model",
        data={"sample": Sample(name="a", count=2)},
    )

    data = event_to_dict(event)["data"]

    assert data == {"sample": {"name": "a", "count": 2}}
    json.dumps(data)


def test_pydantic_model_with_non_json_types_is_reconverted():
    class Stamped(BaseModel):
        when: datetime
        where: Path

    event = Event(
        type=EventType.CYCLE,
        message="m",
        data={"s": Stamped(when=datetime(2026, 1, 2, tzinfo=timezone.utc), where=Path("/w"))},
    )

    assert event_to_dict(event)["data"]["s"] == {
        "when": "2026-01-02T00:00:00+00:00",
        "where": "/w",
    }


def test_cyclic_structure_degrades_to_repr_instead_of_recursing():
    """Untrusted model data can be self-referential; the emitter must survive it."""
    cyclic: dict[str, Any] = {"name": "loop"}
    cyclic["self"] = cyclic
    event = Event(type=EventType.LOG, message="cyclic", data=cyclic)

    converted = event_to_dict(event)["data"]

    assert converted["name"] == "loop"
    # The chain unwinds until the depth guard replaces the innermost value with repr.
    node = converted["self"]
    levels = 0
    while isinstance(node, dict):
        node = node["self"]
        levels += 1
        assert levels <= 20, "depth guard did not stop the cycle"
    assert isinstance(node, str)
    assert "self" in node
    json.dumps(converted)


def test_self_referential_list_degrades_to_repr():
    items: list[Any] = [1]
    items.append(items)
    event = Event(type=EventType.LOG, message="cyclic list", data={"items": items})

    converted = event_to_dict(event)["data"]["items"]

    assert converted[0] == 1
    node = converted[1]
    levels = 0
    while isinstance(node, list):
        node = node[1]
        levels += 1
        assert levels <= 20, "depth guard did not stop the cycle"
    assert isinstance(node, str)
    json.dumps(converted)


def test_self_referential_dataclass_degrades_without_recursion_error():
    @dataclass
    class Node:
        name: str
        child: Node | None = None

    node = Node("a")
    node.child = node
    event = Event(type=EventType.PLAN, message="tree", data={"node": node})

    converted = event_to_dict(event)["data"]["node"]

    assert converted["name"] == "a"
    json.dumps(converted)


def test_deeply_nested_structure_is_cut_off_rather_than_expanded_forever():
    payload: dict[str, Any] = {"leaf": True}
    for _ in range(60):
        payload = {"next": payload}
    event = Event(type=EventType.LOG, message="deep", data=payload)

    converted = event_to_dict(event)["data"]

    depth = 0
    while isinstance(converted, dict) and "next" in converted:
        converted = converted["next"]
        depth += 1

    assert isinstance(converted, str)
    assert 0 < depth <= 15
    json.dumps(event_to_dict(event))
