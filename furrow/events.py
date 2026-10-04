from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"
    PLAN = "plan"
    TASK_START = "task_start"
    TASK_END = "task_end"
    TOOL_CALL = "tool_call"
    TEST_RESULT = "test_result"
    LOG = "log"
    ERROR = "error"


@dataclass(slots=True)
class Event:
    type: EventType
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the web API / websocket payloads.

        POLISH (wave 5b): "data" is a shallow copy. It used to alias `self.data`, so a
        consumer mutating the frame (the web layer hands these straight to a socket)
        silently rewrote the stored event. Payloads are shallow by construction, so a
        `dict(...)` copy is enough; nested values are still shared by design.
        """
        return {
            "type": self.type.value,
            "message": self.message,
            "data": dict(self.data),
            "cycle": self.cycle,
            "timestamp": self.timestamp,
        }


class EventBus:
    """Dependency-free in-process pub/sub bus for live agent activity."""

    def __init__(self, history: int = 500) -> None:
        self._history: deque[Event] = deque(maxlen=max(1, history))
        self._subscribers: list[asyncio.Queue[Event]] = []

    def emit(self, type: EventType, message: str = "", *, cycle: int = 0, **data: Any) -> Event:
        """Record an event and fan it out to subscribers. Never raises."""
        event = Event(type=type, message=message, data=data, cycle=cycle)
        self._history.append(event)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except Exception:
                continue
        return event

    def subscribe(self) -> asyncio.Queue[Event]:
        """Register and return a new queue. History is not replayed - read it via `history()`."""
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=1000)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        """Deregister a queue. Safe to call more than once."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def history(self) -> list[Event]:
        """Return buffered events, oldest first."""
        return list(self._history)

    def clear(self) -> None:
        """Empty the history buffer. Subscribers are untouched."""
        self._history.clear()

    async def stream(self, queue: asyncio.Queue[Event] | None = None) -> AsyncIterator[Event]:
        """Yield events from a queue, managing subscription lifetime when we own the queue."""
        owned = queue is None
        if owned:
            queue = self.subscribe()
        assert queue is not None
        try:
            while True:
                yield await queue.get()
        finally:
            if owned:
                self.unsubscribe(queue)
