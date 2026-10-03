"""Tests for the durable run state, the event record and the in-process bus.

Everything here is offline: no network, no subprocess, no sleeps that outlive a
single event loop tick.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import TaskModel, TaskStatus
from furrow.core.events import Event, EventBus
from furrow.state import RunState, StateStore


def _populated_state() -> RunState:
    """A state object that exercises every field the store has to survive."""
    return RunState(
        goal="build the thing",
        fix_instructions="tests/test_a.py::test_b failed",
        cycles=3,
        tasks=[
            TaskModel(
                id="t1",
                description="first task",
                status=TaskStatus.COMPLETED,
                result="ok",
                attempts=1,
            ),
            TaskModel(
                id="t2",
                description="second task",
                status=TaskStatus.FAILED,
                result="ValueError: boom",
                attempts=2,
            ),
        ],
        done=False,
        halted_reason="max_cycles reached",
        history=[
            {
                "cycle": 1,
                "passed": True,
                "summary": "green",
                "failures": [],
                "task_statuses": {"t1": "completed"},
            },
            {
                "cycle": 2,
                "passed": False,
                "summary": "red",
                "failures": ["ValueError: boom"],
                "task_statuses": {"t2": "failed"},
            },
        ],
    )


def test_save_then_load_round_trips_a_populated_state(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    original = _populated_state()

    store.save(original)
    loaded = store.load()

    assert loaded is not None
    assert loaded == original
    assert loaded.goal == "build the thing"
    assert loaded.halted_reason == "max_cycles reached"
    assert [task.status for task in loaded.tasks] == [TaskStatus.COMPLETED, TaskStatus.FAILED]
    assert loaded.tasks[1].attempts == 2
    assert len(loaded.history) == 2
    assert loaded.history[1]["failures"] == ["ValueError: boom"]


def test_load_returns_none_for_a_missing_file(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "nope" / "state.json")

    assert store.load() is None


def test_load_returns_none_for_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    store = StateStore(path)

    path.write_text("this is not json {{{", encoding="utf-8")
    assert store.load() is None

    # Valid JSON that does not match the schema is also "no state", never a crash.
    path.write_text('{"goal": 42}', encoding="utf-8")
    assert store.load() is None

    # A file that is not even decodable as text.
    path.write_bytes(b"\xff\xfe\x00binary garbage")
    assert store.load() is None


def test_save_creates_parents_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / ".furrow" / "state.json"
    store = StateStore(path)

    store.save(_populated_state())
    store.save(_populated_state())  # a second save must not strand the first temp file

    assert path.is_file()
    assert [entry.name for entry in sorted(path.parent.iterdir())] == ["state.json"]


def test_clear_removes_the_file_and_load_is_none(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    store = StateStore(path)
    store.save(_populated_state())

    store.clear()

    assert not path.exists()
    assert store.load() is None
    store.clear()  # clearing twice is not an error


def test_event_as_dict_is_json_serializable_with_the_documented_keys() -> None:
    event = Event(kind="test_result", cycle=2, message="red", data={"passed": False, "n": 1})

    payload = event.as_dict()

    assert set(payload) == {"kind", "cycle", "message", "data"}
    assert payload["kind"] == "test_result"
    assert payload["cycle"] == 2
    assert json.loads(json.dumps(payload)) == payload
    # as_dict copies the data mapping: mutating the event afterwards cannot rewrite history.
    event.data["passed"] = True
    assert payload["data"]["passed"] is False


async def test_each_subscribe_gets_an_independent_queue() -> None:
    bus = EventBus()

    first = bus.subscribe()
    second = bus.subscribe()

    assert first is not second
    assert bus.subscribers == [first, second]
    bus.emit(Event(kind="notice", message="hello"))
    assert first.qsize() == 1
    assert second.qsize() == 1

    first.get_nowait()
    assert first.qsize() == 0
    assert second.qsize() == 1


async def test_emit_fans_out_and_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    staying = bus.subscribe()
    leaving = bus.subscribe()

    bus.emit(Event(kind="run_start"))
    assert [queue.qsize() for queue in (staying, leaving)] == [1, 1]

    bus.unsubscribe(leaving)
    bus.emit(Event(kind="run_end"))

    assert staying.qsize() == 2
    assert leaving.qsize() == 1
    assert bus.subscribers == [staying]
    bus.unsubscribe(leaving)  # unsubscribing twice is not an error


async def test_close_clears_subscribers_but_leaves_queues_usable() -> None:
    bus = EventBus()
    queue = bus.subscribe()

    bus.close()

    assert bus.subscribers == []
    queue.put_nowait(Event(kind="notice", message="after close"))
    assert queue.get_nowait().message == "after close"
    bus.emit(Event(kind="run_end"))  # no subscribers left: a no-op, not a crash


async def test_a_full_queue_never_blocks_or_raises_the_producer() -> None:
    bus = EventBus(maxsize=3)
    slow = bus.subscribe()

    for index in range(200):
        bus.emit(Event(kind="notice", cycle=index, message=f"event {index}"))

    assert slow.qsize() == 3
    drained = [slow.get_nowait() for _ in range(3)]
    # The newest events survive: a full queue drops the oldest, not the newest.
    assert drained[-1].message == "event 199"
    assert [event.message for event in drained] == ["event 197", "event 198", "event 199"]


async def test_a_slow_consumer_does_not_starve_a_healthy_one() -> None:
    bus = EventBus(maxsize=2)
    slow = bus.subscribe()
    healthy = bus.subscribe()

    for index in range(50):
        bus.emit(Event(kind="notice", message=f"e{index}"))

    assert slow.qsize() == 2
    assert healthy.qsize() == 2
    assert [healthy.get_nowait().message for _ in range(2)] == ["e48", "e49"]


def test_task_status_is_a_str_enum() -> None:
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"
    assert isinstance(TaskStatus.PENDING, str)
    assert TaskStatus("running") is TaskStatus.RUNNING
    assert TaskStatus.RUNNING.value == "running"


def test_task_model_coerces_status_assignment_into_the_enum() -> None:
    task = TaskModel(id="t1", description="do it")

    task.status = "completed"

    assert task.status is TaskStatus.COMPLETED
    assert task.status == "completed"
    # validate_assignment also rejects nonsense instead of storing it verbatim.
    with pytest.raises(ValidationError):
        task.status = "nonsense"


async def test_queue_round_trip_of_an_event_from_the_bus() -> None:
    bus = EventBus()
    queue: asyncio.Queue[Event] = bus.subscribe()

    bus.emit(Event(kind="cycle_start", cycle=1))

    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event.kind == "cycle_start"
    assert event.as_dict() == {"kind": "cycle_start", "cycle": 1, "message": "", "data": {}}