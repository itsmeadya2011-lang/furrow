"""Event bus used to stream orchestrator progress to CLI and web consumers."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any

MAX_QUEUE_SIZE = 1000


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


@dataclass
class EventBusStats:
    """Statistics for the event bus."""
    total_emitted: int = 0
    total_dropped: int = 0
    by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dropped_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    subscriber_count: int = 0
    avg_queue_depth: float = 0.0


class EventBus:
    """Fan-out bus: subscribers each get their own bounded queue.

    Every subscriber queue holds at most ``max_queue_size`` events. When a
    subscriber stops draining, the oldest queued event is dropped to make room
    for the newest one, so a slow consumer still sees the most recent events.
    """

    def __init__(self, max_queue_size: int = 1000) -> None:
        self._max_queue_size = max_queue_size
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._dropped = 0
        self._dropped_by_queue: dict[asyncio.Queue[Event], int] = {}
        self._stats_lock = Lock()
        self._total_emitted = 0
        self._by_type: dict[str, int] = defaultdict(int)
        self._dropped_by_type: dict[str, int] = defaultdict(int)
        self._emit_latencies: list[float] = []

    def subscribe(self) -> asyncio.Queue[Event]:
        """Register and return a new bounded queue that receives every emitted event."""
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        """Remove a previously registered queue. Removing an unknown queue is a no-op."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)
        self._dropped_by_queue.pop(queue, None)

    def emit(self, event: Event) -> None:
        """Publish an event to every subscriber. Never blocks and never raises."""
        start = time.monotonic()
        event_type = event.type.value if isinstance(event.type, EventType) else str(event.type)
        with self._stats_lock:
            self._total_emitted += 1
            self._by_type[event_type] += 1
        for queue in list(self._subscribers):
            self._deliver(queue, event)
        latency = time.monotonic() - start
        with self._stats_lock:
            self._emit_latencies.append(latency)
            if len(self._emit_latencies) > 1000:
                self._emit_latencies = self._emit_latencies[-1000:]

    def _deliver(self, queue: asyncio.Queue[Event], event: Event) -> None:
        """Put ``event`` on ``queue``, dropping the oldest event if it is full."""
        event_type = event.type.value if isinstance(event.type, EventType) else str(event.type)
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
            with self._stats_lock:
                self._dropped += 1
                self._dropped_by_queue[queue] = self._dropped_by_queue.get(queue, 0) + 1
                self._dropped_by_type[event_type] += 1
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

    def get_stats(self) -> EventBusStats:
        """Get current event bus statistics."""
        with self._stats_lock:
            total_queue_depth = sum(q.qsize() for q in self._subscribers)
            avg_queue_depth = (
                total_queue_depth / len(self._subscribers) if self._subscribers else 0.0
            )
            return EventBusStats(
                total_emitted=self._total_emitted,
                total_dropped=self._dropped,
                by_type=dict(self._by_type),
                dropped_by_type=dict(self._dropped_by_type),
                subscriber_count=len(self._subscribers),
                avg_queue_depth=avg_queue_depth,
            )

    def reset_stats(self) -> None:
        """Reset event bus statistics."""
        with self._stats_lock:
            self._total_emitted = 0
            self._by_type.clear()
            self._dropped = 0
            self._dropped_by_type.clear()
            self._dropped_by_queue.clear()
            self._emit_latencies.clear()