from __future__ import annotations

from collections.abc import Callable


class EventBus:
    """Simple synchronous pub/sub for orchestrator events (dicts with a 'type' key)."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[dict], None]] = []

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[dict], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def emit(self, event: dict) -> None:
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception:  # pragma: no cover  # noqa: BLE001, S110
                pass