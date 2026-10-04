from __future__ import annotations

import contextlib
import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Enumeration of Furrow loop event types."""

    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    CYCLE_STARTED = "cycle.started"
    PLAN_CREATED = "plan.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_SKIPPED = "task.skipped"
    TESTS_COMPLETED = "tests.completed"
    CYCLE_COMPLETED = "cycle.completed"
    LOG = "log"


@dataclass(slots=True)
class Event:
    """A single point-in-time occurrence in the Furrow loop.

    All values in `data` must be JSON-safe — that is the caller's
    responsibility.
    """

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict: type as its string value, plus cycle,
        timestamp, and data."""
        return {
            "type": self.type.value,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "data": self.data,
        }


class EventBus:
    """Fan-out bus for loop events. Handlers may be sync or async; handler
    exceptions are logged and swallowed so one bad subscriber cannot break the
    loop."""

    def __init__(self) -> None:
        self._handlers: list[Callable[[Event], Any]] = []

    def subscribe(self, handler: Callable[[Event], Any]) -> Callable[[], None]:
        """Register handler; return a zero-arg unsubscribe callable."""
        self._handlers.append(handler)

        def _unsubscribe() -> None:
            self.unsubscribe(handler)

        return _unsubscribe

    def unsubscribe(self, handler: Callable[[Event], Any]) -> None:
        """Remove a previously registered handler."""
        with contextlib.suppress(ValueError):
            self._handlers.remove(handler)

    @property
    def handler_count(self) -> int:
        """Number of currently registered handlers."""
        return len(self._handlers)

    async def emit(self, event: Event) -> None:
        """Deliver to every handler. Coroutine handlers — and sync handlers that
        return an awaitable — are awaited. Per-handler exceptions are caught and
        swallowed (logged with `logging.exception`). Iteration is over a copy of
        the handler list so handlers may subscribe/unsubscribe during dispatch.
        """
        for handler in list(self._handlers):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Event handler failed for %s", event.type)

    def emit_nowait(self, event: Event) -> None:
        """Synchronous dispatch for sync-only handlers. Skip (do not run)
        coroutine functions, and log a warning if a subscriber is async."""
        for handler in list(self._handlers):
            if inspect.iscoroutinefunction(handler):
                logger.warning(
                    "Skipping async handler %s for sync-only emit on %s",
                    handler,
                    event.type,
                )
            else:
                try:
                    result = handler(event)
                    if inspect.isawaitable(result):
                        logger.warning(
                            "Handler %s returned an awaitable; sync-only emit cannot "
                            "await it on %s",
                            handler,
                            event.type,
                        )
                        if inspect.iscoroutine(result):
                            result.close()
                except Exception:
                    logger.exception("Sync event handler failed for %s", event.type)


# ── Factory functions ─────────────────────────────────────────────────────────


def run_started(goal: str, *, max_cycles: int = 0) -> Event:
    """Emitted once when the orchestrator starts a run."""
    return Event(
        type=EventType.RUN_STARTED,
        data={"goal": goal, "max_cycles": max_cycles},
    )


def run_completed(cycles: int, reason: str) -> Event:
    """Emitted after the run exits normally."""
    return Event(
        type=EventType.RUN_COMPLETED,
        data={"cycles": cycles, "reason": reason},
    )


def run_failed(error: str) -> Event:
    """Emitted if the run aborts due to an unrecoverable error."""
    return Event(
        type=EventType.RUN_FAILED,
        data={"error": error},
    )


def cycle_started(cycle: int) -> Event:
    """Emitted at the start of each development cycle."""
    return Event(
        type=EventType.CYCLE_STARTED,
        data={"cycle": cycle},
        cycle=cycle,
    )


def plan_created(plan: dict[str, Any], *, cycle: int = 0) -> Event:
    """Emitted after the planner produces a task plan.

    ``cycle`` is keyword-only and defaults to 0 so callers that predate it keep
    working; every per-cycle event factory takes the cycle explicitly, and
    consumers that group the stream by cycle (the web dashboard) need the plan
    to sit in the cycle that produced it.
    """
    return Event(
        type=EventType.PLAN_CREATED,
        data={"plan": plan},
        cycle=cycle,
    )


def task_started(
    task_id: str,
    description: str,
    files: list[str],
    *,
    cycle: int = 0,
) -> Event:
    """Emitted when a worker begins a task."""
    return Event(
        type=EventType.TASK_STARTED,
        data={
            "task_id": task_id,
            "description": description,
            "files": files,
        },
        cycle=cycle,
    )


def task_completed(
    task_id: str,
    result: str,
    duration: float = 0.0,
    *,
    cycle: int = 0,
) -> Event:
    """Emitted after a task finishes successfully."""
    return Event(
        type=EventType.TASK_COMPLETED,
        data={
            "task_id": task_id,
            "result": result,
            "duration": duration,
        },
        cycle=cycle,
    )


def task_failed(
    task_id: str,
    error: str,
    duration: float = 0.0,
    *,
    cycle: int = 0,
) -> Event:
    """Emitted after a task ends in failure."""
    return Event(
        type=EventType.TASK_FAILED,
        data={
            "task_id": task_id,
            "error": error,
            "duration": duration,
        },
        cycle=cycle,
    )


def task_skipped(task_id: str, reason: str, *, cycle: int = 0) -> Event:
    """Emitted when a task is skipped."""
    return Event(
        type=EventType.TASK_SKIPPED,
        data={
            "task_id": task_id,
            "reason": reason,
        },
        cycle=cycle,
    )


def tests_completed(
    passed: bool,
    summary: str,
    failures: list[str],
    *,
    cycle: int = 0,
) -> Event:
    """Emitted after the test runner finishes."""
    return Event(
        type=EventType.TESTS_COMPLETED,
        data={
            "passed": passed,
            "summary": summary,
            "failures": failures,
        },
        cycle=cycle,
    )


def cycle_completed(cycle: int, passed: bool) -> Event:
    """Emitted after all tasks and tests for a cycle finish."""
    return Event(
        type=EventType.CYCLE_COMPLETED,
        data={"passed": passed},
        cycle=cycle,
    )


def log(message: str, level: str = "info", *, cycle: int = 0) -> Event:
    """Emitted for generic log messages."""
    return Event(
        type=EventType.LOG,
        data={"message": message, "level": level},
        cycle=cycle,
    )
