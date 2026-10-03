"""Tiny in-process pub/sub used to observe an orchestrator run.

The bus is deliberately dependency free: the CLI, the web socket and tests all
consume the same event stream, and nothing here touches the network or disk.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

EventKind = Literal[
    "run_start",
    "cycle_start",
    "plan",
    "task_start",
    "task_end",
    "test_result",
    "cycle_end",
    "run_end",
    "error",
    "notice",
]

DEFAULT_QUEUE_SIZE = 1000


@dataclass
class Event:
    """A single observation emitted by :class:`furrow.core.orchestrator.Orchestrator`."""

    kind: EventKind
    cycle: int = 0
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the event."""
        return {
            "kind": self.kind,
            "cycle": self.cycle,
            "message": self.message,
            "data": dict(self.data),
        }


class EventBus:
    """Fan-out event bus. Every :meth:`subscribe` call yields its own queue."""

    def __init__(self, maxsize: int = DEFAULT_QUEUE_SIZE) -> None:
        self._maxsize = max(1, maxsize)
        self._subscribers: list[asyncio.Queue[Event]] = []

    @property
    def subscribers(self) -> list[asyncio.Queue[Event]]:
        """Live view of the currently subscribed queues."""
        return list(self._subscribers)

    def subscribe(self) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            return

    def emit(self, event: Event) -> None:
        """Publish ``event`` to every subscriber. Never raises, never blocks."""
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover - racy drain
                    pass
            except Exception:  # pragma: no cover - defensive: a bad sink must not stop the loop
                continue

    def close(self) -> None:
        """Drop all subscribers. Queues themselves stay usable."""
        self._subscribers.clear()