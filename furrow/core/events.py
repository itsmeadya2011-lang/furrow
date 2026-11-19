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

    # Run lifecycle
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"
    
    # Cycle lifecycle
    CYCLE_STARTED = "cycle_started"
    CYCLE_FINISHED = "cycle_finished"
    
    # Planning
    PLAN_CREATED = "plan_created"
    PLAN_FAILED = "plan_failed"
    
    # Task lifecycle
    TASK_STARTED = "task_started"
    TASK_FINISHED = "task_finished"
    TASK_FAILED = "task_failed"
    TASK_RETRYING = "task_retrying"
    TASK_SKIPPED = "task_skipped"
    TASK_BLOCKED = "task_blocked"
    
    # Test lifecycle
    TEST_STARTED = "test_started"
    TEST_FINISHED = "test_finished"
    TEST_FAILED = "test_failed"
    TEST_SKIPPED = "test_skipped"
    
    # State persistence
    STATE_SAVED = "state_saved"
    STATE_LOAD_FAILED = "state_load_failed"
    STATE_SAVE_FAILED = "state_save_failed"
    
    # LLM interactions
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_ERROR = "llm_error"
    
    # Tool usage
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    
    # Command execution
    COMMAND_STARTED = "command_started"
    COMMAND_FINISHED = "command_finished"
    COMMAND_FAILED = "command_failed"
    COMMAND_DENIED = "command_denied"
    
    # Progress tracking
    PROGRESS_UPDATE = "progress_update"
    NO_PROGRESS_DETECTED = "no_progress_detected"
    
    # Cost tracking
    COST_UPDATE = "cost_update"
    
    # General
    LOG = "log"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Event:
    """A single progress event."""

    type: EventType
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    trace_id: str = ""
    span_id: str = ""
    cycle: int | None = None
    task_id: str = ""

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
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "cycle": self.cycle,
            "task_id": self.task_id,
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


# Convenience functions for creating common events
def create_run_started_event(goal: str, trace_id: str = "") -> Event:
    return Event(type=EventType.RUN_STARTED, message=f"Run started: {goal[:100]}", 
                 data={"goal": goal}, trace_id=trace_id)


def create_run_finished_event(result: Any, trace_id: str = "") -> Event:
    return Event(
        type=EventType.RUN_FINISHED,
        message=f"Run finished: {result.stopped_reason}",
        data={
            "passed": result.passed,
            "cycles": result.cycles,
            "tasks_completed": result.tasks_completed,
            "tasks_failed": result.tasks_failed,
            "stopped_reason": result.stopped_reason,
        },
        trace_id=trace_id,
    )


def create_task_started_event(task_id: str, description: str, cycle: int, trace_id: str = "", span_id: str = "") -> Event:
    return Event(
        type=EventType.TASK_STARTED,
        message=f"Task {task_id} started: {description[:100]}",
        data={"task_id": task_id, "description": description},
        trace_id=trace_id,
        span_id=span_id,
        cycle=cycle,
        task_id=task_id,
    )


def create_task_finished_event(task_id: str, success: bool, summary: str, cycle: int, trace_id: str = "", span_id: str = "") -> Event:
    return Event(
        type=EventType.TASK_FINISHED if success else EventType.TASK_FAILED,
        message=f"Task {task_id} {'completed' if success else 'failed'}: {summary[:100]}",
        data={"task_id": task_id, "success": success, "summary": summary},
        trace_id=trace_id,
        span_id=span_id,
        cycle=cycle,
        task_id=task_id,
    )


def create_llm_request_event(model: str, operation: str, trace_id: str = "", span_id: str = "") -> Event:
    return Event(
        type=EventType.LLM_REQUEST,
        message=f"LLM {operation} request to {model}",
        data={"model": model, "operation": operation},
        trace_id=trace_id,
        span_id=span_id,
    )


def create_llm_response_event(model: str, operation: str, usage: Any, trace_id: str = "", span_id: str = "") -> Event:
    return Event(
        type=EventType.LLM_RESPONSE,
        message=f"LLM {operation} response from {model}",
        data={
            "model": model,
            "operation": operation,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else str(usage),
        },
        trace_id=trace_id,
        span_id=span_id,
    )


def create_command_event(event_type: EventType, command: str, task_id: str, cycle: int, trace_id: str = "", span_id: str = "") -> Event:
    return Event(
        type=event_type,
        message=f"Command {event_type.value}: {command[:100]}",
        data={"command": command, "task_id": task_id},
        trace_id=trace_id,
        span_id=span_id,
        cycle=cycle,
        task_id=task_id,
    )


def create_cost_update_event(cost_usd: float, total_tokens: int, trace_id: str = "") -> Event:
    return Event(
        type=EventType.COST_UPDATE,
        message=f"Cost update: ${cost_usd:.6f}, {total_tokens} tokens",
        data={"cost_usd": cost_usd, "total_tokens": total_tokens},
        trace_id=trace_id,
    )


def create_progress_event(message: str, cycle: int, trace_id: str = "") -> Event:
    return Event(
        type=EventType.PROGRESS_UPDATE,
        message=message,
        data={},
        trace_id=trace_id,
        cycle=cycle,
    )
