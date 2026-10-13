from __future__ import annotations

from furrow.core.events import EventBus


def test_subscribe_and_emit():
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe(received.append)
    bus.emit({"type": "foo", "payload": 1})
    assert received == [{"type": "foo", "payload": 1}]


def test_emit_with_no_subscribers_is_noop():
    bus = EventBus()
    bus.emit({"type": "whatever"})


def test_unsubscribe_stops_receiving():
    bus = EventBus()
    received: list[dict] = []

    def cb(event: dict) -> None:
        received.append(event)

    bus.subscribe(cb)
    bus.emit({"type": "a"})
    bus.unsubscribe(cb)
    bus.emit({"type": "b"})
    assert received == [{"type": "a"}]


def test_unsubscribe_unknown_callback_does_not_raise():
    bus = EventBus()
    bus.unsubscribe(lambda e: None)


def test_multiple_subscribers_called_in_order():
    bus = EventBus()
    order: list[str] = []

    def cb_a(event: dict) -> None:
        order.append("a")

    def cb_b(event: dict) -> None:
        order.append("b")

    bus.subscribe(cb_a)
    bus.subscribe(cb_b)
    bus.emit({"type": "x"})
    assert order == ["a", "b"]


def test_exception_in_callback_does_not_break_others():
    bus = EventBus()
    received: list[dict] = []

    def bad(event: dict) -> None:
        raise RuntimeError("boom")

    def good(event: dict) -> None:
        received.append(event)

    bus.subscribe(bad)
    bus.subscribe(good)
    bus.subscribe(bad)
    bus.emit({"type": "safe"})

    assert received == [{"type": "safe"}]


def test_exception_in_callback_does_not_break_emit():
    bus = EventBus()
    bus.subscribe(lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.emit({"type": "safe"})  # must not raise


def test_unsubscribe_only_removes_one_instance():
    bus = EventBus()
    count = 0

    def cb(event: dict) -> None:
        nonlocal count
        count += 1

    bus.subscribe(cb)
    bus.subscribe(cb)
    bus.unsubscribe(cb)
    bus.emit({"type": "x"})
    assert count == 1


def test_subscribed_callback_receives_same_event_object():
    bus = EventBus()
    captured: list[dict] = []
    bus.subscribe(captured.append)
    event = {"type": "evt", "data": [1, 2, 3]}
    bus.emit(event)
    assert captured[0] is event
