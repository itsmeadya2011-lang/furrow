"""Async, in-process event bus used to stream orchestrator progress.

The orchestrator publishes events here; the CLI renderer and the web websocket
subscribe and forward them to the user.  The module is intentionally tiny and
dependency free (stdlib + :mod:`asyncio` only) so that it can be imported from
anywhere in the package.  In particular it must never import
:mod:`furrow.core.orchestrator` -- that would create an import cycle because the
orchestrator publishes events of this type.

Design rules:

* :meth:`EventBus.publish` never raises and never blocks.  Instrumentation must
  not be able to kill a run, and a slow consumer must not stall the loop.
* A subscriber that cannot keep up loses its *oldest* queued events (the newest
  ones are the interesting ones) instead of applying back-pressure.
* Every subscription is independent and closing one only affects that consumer.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only (pydantic dependency, py<3.11)
    from typing_extensions import Self

logger = logging.getLogger(__name__)

__all__ = ["Event", "EventBus", "EventKind", "json_safe"]


class EventKind(str, Enum):
    """The kinds of progress event an orchestrator run can emit."""

    RUN_START = "run_start"
    CYCLE_START = "cycle_start"
    PLAN = "plan"
    TASK_START = "task_start"
    TASK_RESULT = "task_result"
    TEST_RESULT = "test_result"
    CYCLE_END = "cycle_end"
    RUN_END = "run_end"
    ERROR = "error"
    LOG = "log"


def json_safe(value: Any) -> Any:
    """Convert ``value`` into something :func:`json.dumps` can always encode.

    Pydantic models, dataclasses, enums, paths and datetimes are converted to
    their natural JSON equivalents; anything else falls back to :func:`repr`.
    Non-finite floats become strings because strict parsers (notably the browser
    ``JSON.parse`` used by the web UI) reject ``NaN`` and ``Infinity``.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    # Duck-typed on purpose: this module must not depend on pydantic.
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        for kwargs in ({"mode": "json"}, {}):
            try:
                return json_safe(dump(**kwargs))
            except Exception:  # a broken dumper must not break the bus
                logger.debug("model_dump(%s) failed", kwargs, exc_info=True)
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return json_safe(asdict(value))
        except Exception:
            logger.debug("asdict() failed for %r", type(value).__name__, exc_info=True)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return json_safe(to_dict())
        except Exception:
            logger.debug("to_dict() failed for %r", type(value).__name__, exc_info=True)
    return repr(value)


@dataclass(slots=True)
class Event:
    """A single progress notification.

    Attributes:
        kind: What happened (see :class:`EventKind`).
        message: Short human readable line for the CLI/web log.
        cycle: Zero-based (or one-based, per caller) cycle counter.
        data: JSON-serialisable payload, e.g. a plan dump or task result.
        timestamp: Epoch seconds captured when the event was created.
    """

    kind: EventKind
    message: str = ""
    cycle: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict for websocket/JSON rendering.

        ``timestamp`` is an ISO-8601 UTC string (what the UI displays) and
        ``ts`` is the raw epoch float (useful for ordering and elapsed times).
        """
        kind = self.kind.value if isinstance(self.kind, EventKind) else str(self.kind)
        try:
            stamp = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, TypeError, ValueError):
            stamp = ""
        return json_safe(
            {
                "kind": kind,
                "message": self.message,
                "cycle": self.cycle,
                "data": self.data,
                "timestamp": stamp,
                "ts": self.timestamp,
            }
        )


class _SubscriptionClosed(RuntimeError):
    """Raised by :meth:`_Subscription.get` once the subscription is closed.

    A plain :class:`RuntimeError` subclass (rather than
    :class:`asyncio.CancelledError`) so that ``async for`` can translate it into
    ``StopAsyncIteration`` without swallowing real task cancellation.
    """


class _Subscription:
    """A single consumer's view of an :class:`EventBus`.

    Use it as an async iterator::

        async with bus.subscribe() as sub:
            async for event in sub:
                if event.kind is EventKind.RUN_END:
                    break
    """

    __slots__ = ("_bus", "_closed", "_queue", "dropped")

    def __init__(self, bus: EventBus, queue: asyncio.Queue[Event | None]) -> None:
        self._bus = bus
        self._queue = queue
        self._closed = False
        self.dropped = 0

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    def _offer(self, event: Event) -> None:
        """Enqueue ``event`` for this consumer without ever blocking."""
        if self._closed:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Slow consumer: drop the oldest event to make room for the newest.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
                self.dropped += 1
            except Exception:  # dropping must never break a run
                logger.debug("dropped event for a slow subscriber", exc_info=True)
        except Exception:
            logger.debug("event fan-out failed", exc_info=True)

    async def get(self) -> Event:
        """Await the next event.

        Raises ``RuntimeError`` (:class:`_SubscriptionClosed`) when the
        subscription is closed.  :class:`asyncio.CancelledError` propagates
        untouched so cancelling a consumer task keeps working normally.
        """
        if self._closed:
            raise _SubscriptionClosed("event subscription is closed")
        event = await self._queue.get()
        if event is None:  # close() sentinel
            raise _SubscriptionClosed("event subscription is closed")
        return event

    def close(self) -> None:
        """Unsubscribe and wake up any consumer currently blocked in :meth:`get`."""
        if self._closed:
            return
        self._closed = True
        self._bus._discard(self)
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except Exception:
                logger.debug("could not wake a closed subscription", exc_info=True)
        except Exception:
            logger.debug("could not wake a closed subscription", exc_info=True)

    def __aiter__(self) -> _Subscription:
        return self

    async def __anext__(self) -> Event:
        try:
            return await self.get()
        except _SubscriptionClosed:
            raise StopAsyncIteration from None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "closed" if self._closed else "open"
        return f"<_Subscription {state} queued={self._queue.qsize()} dropped={self.dropped}>"


class EventBus:
    """Fan-out event bus with a bounded replay buffer.

    Args:
        history_limit: Maximum number of events kept for replay / ``recent``.
        queue_size: Per-subscriber buffer; slow subscribers drop the oldest
            events rather than blocking the publisher.
    """

    def __init__(self, history_limit: int = 1000, queue_size: int = 256) -> None:
        self.history: deque[Event] = deque(maxlen=max(1, history_limit))
        self.history_limit = max(1, history_limit)
        self.queue_size = max(1, queue_size)
        self._subscribers: set[_Subscription] = set()

    async def publish(self, event: Event) -> None:
        """Record ``event`` in history and fan it out to every subscriber.

        Never raises: a broken subscriber must not be able to fail a run.
        """
        try:
            self.history.append(event)
            for sub in list(self._subscribers):
                sub._offer(event)
        except Exception:
            logger.debug("event bus publish failed", exc_info=True)

    def subscribe(self) -> _Subscription:
        """Register and return a new subscription."""
        sub = _Subscription(self, asyncio.Queue(maxsize=self.queue_size))
        self._subscribers.add(sub)
        return sub

    def _discard(self, sub: _Subscription) -> None:
        """Forget ``sub`` (called by :meth:`_Subscription.close`)."""
        self._subscribers.discard(sub)

    def subscriber_count(self) -> int:
        """Number of currently open subscriptions."""
        return len(self._subscribers)

    def close_all(self) -> None:
        """Close every open subscription, waking up blocked consumers."""
        for sub in list(self._subscribers):
            sub.close()
        self._subscribers.clear()

    def recent(self, limit: int = 50) -> list[Event]:
        """Return up to ``limit`` most recent events, oldest first."""
        if limit <= 0:
            return []
        items = list(self.history)
        return items[-limit:]

    async def wait_for(self, kind: EventKind, timeout: float = 5.0) -> Event:
        """Return the first event of ``kind``, from history or from the future.

        Handy in tests and in the CLI when it needs to block until a particular
        milestone (``RUN_END``, ``TEST_RESULT``, ...) shows up.

        Raises:
            TimeoutError: if no matching event arrives within ``timeout``.
        """
        label = getattr(kind, "value", kind)
        for event in reversed(self.history):
            if event.kind == kind:
                return event

        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        sub = self.subscribe()
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError(f"no {label!r} event within {timeout}s")
                try:
                    event = await asyncio.wait_for(sub.get(), remaining)
                except (asyncio.TimeoutError, TimeoutError) as exc:
                    raise TimeoutError(f"no {label!r} event within {timeout}s") from exc
                if event.kind == kind:
                    return event
        finally:
            sub.close()
