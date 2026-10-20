"""Event bus used to stream orchestrator progress to CLI and web consumers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Lifecycle events emitted by the orchestrator."""

    CYCLE_STARTED = "cycle_started"
    CYCLE_FINISHED = "cycle_finished"
    PLAN_CREATED = "plan_created"
    TASK_STARTED = "task_started"
    TASK_FINISHED = "task_finished"
    TEST_STARTED = "test_started"
    TEST_FINISHED = "test_finished"
    STATE_SAVED = "state_saved"
    LOG = "log"
    ERROR = "error"


@dataclass
class Event:
    """A single progress event."""

    type: EventType
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the event."""
        return {
            "type": self.type.value if isinstance(self.type, EventType) else str(self.type),
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class EventBus:
    """Fan-out bus: subscribers each get their own unbounded queue."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []

    def subscribe(self) -> asyncio.Queue[Event]:
        """Register and return a new queue that receives every emitted event."""
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        """Remove a previously registered queue. Removing an unknown queue is a no-op."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def emit(self, event: Event) -> None:
        """Publish an event to every subscriber. Never blocks and never raises."""
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - queues are unbounded
                continue
            except Exception:  # pragma: no cover - defensive, one bad subscriber
                continue

    async def emit_async(self, event: Event) -> None:
        """Async variant of :meth:`emit`."""
        self.emit(event)

    def clear(self) -> None:
        """Drop every subscriber."""
        self._subscribers.clear()

    @property
    def subscriber_count(self) -> int:
        """Number of currently registered subscribers."""
        return len(self._subscribers)
