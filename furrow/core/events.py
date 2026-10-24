"""Event bus used to stream orchestrator progress to CLI and web consumers."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

#: Maximum number of events buffered for a single subscriber. A subscriber that
#: stops draining loses its *oldest* events rather than growing without bound,
#: so the queue always holds the most recent events. The value is far larger
#: than the number of events a normal run emits, so healthy subscribers never
#: lose anything.
MAX_QUEUE_SIZE = 1000


class EventType(str, Enum):
    """Lifecycle events emitted by the orchestrator."""

    CYCLE_STARTED = "cycle_started"
    CYCLE_FINISHED = "cycle_finished"
    DONE = "done"
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
    """Fan-out bus: subscribers each get their own bounded queue.

    Every subscriber queue holds at most :data:`MAX_QUEUE_SIZE` events. When a
    subscriber stops draining, the oldest queued event is dropped to make room
    for the newest one, so a slow consumer still sees the most recent events.
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._dropped = 0
        self._dropped_by_queue: dict[asyncio.Queue[Event], int] = {}

    def subscribe(self) -> asyncio.Queue[Event]:
        """Register and return a new bounded queue that receives every emitted event."""
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        """Remove a previously registered queue. Removing an unknown queue is a no-op."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)
        self._dropped_by_queue.pop(queue, None)

    def emit(self, event: Event) -> None:
        """Publish an event to every subscriber. Never blocks and never raises."""
        for queue in list(self._subscribers):
            self._deliver(queue, event)

    def _deliver(self, queue: asyncio.Queue[Event], event: Event) -> None:
        """Put ``event`` on ``queue``, dropping the oldest event if it is full."""
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            queue.put_nowait(event)
            return
        # The subscriber is not draining: drop the oldest queued event so that
        # the newest event is always delivered.
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - raced with a consumer
            pass
        except Exception:  # pragma: no cover - defensive, one bad subscriber
            return
        else:
            self._dropped += 1
            self._dropped_by_queue[queue] = self._dropped_by_queue.get(queue, 0) + 1
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            queue.put_nowait(event)

    async def emit_async(self, event: Event) -> None:
        """Async variant of :meth:`emit`."""
        self.emit(event)

    def clear(self) -> None:
        """Drop every subscriber."""
        self._subscribers.clear()
        self._dropped_by_queue.clear()

    @property
    def subscriber_count(self) -> int:
        """Number of currently registered subscribers."""
        return len(self._subscribers)

    @property
    def dropped_count(self) -> int:
        """Total number of events dropped because a subscriber was not draining.

        The counter is cumulative for the lifetime of the bus and is not reset
        by :meth:`clear`.
        """
        return self._dropped

    def dropped_for(self, queue: asyncio.Queue[Event]) -> int:
        """Number of events dropped for a single subscriber queue."""
        return self._dropped_by_queue.get(queue, 0)
