"""UI-facing progress stream for a furrow run.

This module is the single contract between the orchestrator (which produces
events) and every consumer of progress: the CLI, the web server, tests and
anything embedding the library.  It is deliberately dependency free -- stdlib
only, no import-time side effects, no import of the rest of ``furrow`` -- so
that any consumer can import it on its own.

Typical use::

    on_event = None  # or a sync/async callable taking an Event
    await emit(on_event, make_event(EventKind.CYCLE, cycle=1))
    print(format_event(event))          # human/CLI rendering
    payload = event.to_dict()           # JSON-safe dict for the web server
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any

__all__ = [
    "Event",
    "EventCallback",
    "EventFormatter",
    "EventKind",
    "emit",
    "format_event",
    "make_event",
]


class EventKind(str, Enum):
    """Kinds of progress events emitted while a run progresses.

    The members are plain ``str`` values so an event kind can be compared with,
    and serialized as, a bare string.  ``DONE`` marks the end of a run whether
    it succeeded or not; ``ERROR`` reports a failure that did not stop the run.
    """

    CYCLE = "cycle"
    PLAN = "plan"
    TASK_START = "task_start"
    TASK_DONE = "task_done"
    TASK_FAILED = "task_failed"
    TASK_SKIPPED = "task_skipped"
    TESTS = "tests"
    LOG = "log"
    DONE = "done"
    ERROR = "error"


@dataclass
class Event:
    """A single progress event.

    Attributes:
        kind: What happened, see :class:`EventKind`.
        message: Free-form short human text about the event.
        cycle: 1-based cycle number the event belongs to (``0`` when unused).
        task_id: Identifier of the task the event is about, if any.
        data: Extra structured payload; rendered by :func:`format_event` for
            a few well-known keys (``files``, ``count``, ``summary``).
        ts: Unix timestamp, captured when the event is constructed.
    """

    kind: EventKind
    message: str = ""
    cycle: int = 0
    task_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Return a fully JSON-serializable view of the event.

        ``kind`` becomes its string value and ``data`` is converted key by key
        (nested enums to strings, containers rebuilt, pydantic-style models
        dumped, anything else stringified), so the result can be handed
        straight to :func:`json.dumps` by the web server.
        """
        return {
            "kind": _kind_value(self.kind),
            "message": self.message,
            "cycle": self.cycle,
            "task_id": self.task_id,
            "data": _json_safe(self.data),
            "ts": self.ts,
        }


EventCallback = Callable[[Event], Awaitable[None] | None]
"""Consumer callback: a sync or async callable receiving one :class:`Event`."""


async def emit(on_event: EventCallback | None, event: Event) -> None:
    """Deliver ``event`` to ``on_event``, awaiting async callbacks.

    Args:
        on_event: Sync or async callback, or ``None`` to drop the event.
        event: The event to deliver.

    A missing callback is a no-op.  Consumer errors are swallowed so a broken
    UI can never take down a running agent; only ``Exception`` is caught, so
    ``asyncio.CancelledError`` (a ``BaseException``) and other
    ``BaseException`` subclasses still propagate and keep cancellation working.
    """
    if on_event is None:
        return
    try:
        result = on_event(event)
        if inspect.isawaitable(result):
            await result
    except Exception:
        return


def make_event(
    kind: EventKind,
    message: str = "",
    *,
    cycle: int = 0,
    task_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> Event:
    """Build an :class:`Event`, normalizing ``data=None`` to an empty dict.

    Args:
        kind: The event kind; a matching string is coerced to :class:`EventKind`.
        message: Optional short human text.
        cycle: 1-based cycle number, if applicable.
        task_id: Task identifier, if applicable.
        data: Optional structured payload; ``None`` becomes ``{}``.

    Returns:
        The new event, timestamped with :func:`time.time`.
    """
    return Event(
        kind=EventKind(kind),
        message=message,
        cycle=cycle,
        task_id=task_id,
        data=dict(data) if data else {},
    )


@dataclass(frozen=True)
class _Template:
    """Rendering recipe: a format string plus its default trailing detail."""

    line: str
    detail: str = ""


_TEMPLATES: dict[EventKind, _Template] = {
    EventKind.CYCLE: _Template("cycle {cycle} · started", "started"),
    EventKind.PLAN: _Template("cycle {cycle} · plan · {count} task(s) planned"),
    EventKind.TASK_START: _Template("task_start {task} · started", "started"),
    EventKind.TASK_DONE: _Template("task_done {task} · {detail}", "done"),
    EventKind.TASK_FAILED: _Template("task_failed {task} · {detail}", "failed"),
    EventKind.TASK_SKIPPED: _Template("task_skipped {task} · {detail}", "skipped"),
    EventKind.TESTS: _Template("tests · {detail}", "finished"),
    EventKind.LOG: _Template("log · {detail}", ""),
    EventKind.DONE: _Template("done · {detail}", "finished"),
    EventKind.ERROR: _Template("error · {detail}", "failed"),
}


def format_event(event: Event) -> str:
    """Render ``event`` as one short single line for humans (e.g. the CLI).

    Args:
        event: The event to render.

    Returns:
        A line such as ``cycle 2 · plan · 3 task(s) planned`` or
        ``task_done t3 · wrote 2 file(s)``.  Kinds without a template, and
        events whose message is the only useful information, fall back to
        ``event.message`` or the kind's string value.
    """
    recipe = _TEMPLATES.get(event.kind)
    if recipe is None:
        return event.message or _kind_value(event.kind)
    values = {
        "cycle": event.cycle,
        "task": _task_label(event.task_id),
        "count": _count(event),
        "detail": _detail(event) or recipe.detail,
    }
    try:
        line = recipe.line.format(**values)
    except (IndexError, KeyError, ValueError):
        return event.message or _kind_value(event.kind)
    return " · ".join(part.strip() for part in line.split(" · ") if part.strip())


class EventFormatter:
    """Callable renderer for events, sharing :func:`format_event`'s templates.

    Instances hold no state; create one when a consumer wants to pass a
    rendering callable around, otherwise call :func:`format_event` directly.
    """

    def format(self, event: Event) -> str:
        """Return the single-line human rendering of ``event``."""
        return format_event(event)

    def __call__(self, event: Event) -> str:
        """Return the single-line human rendering of ``event``."""
        return self.format(event)


def _kind_value(kind: EventKind | str) -> str:
    """Return the string value of ``kind``, tolerating a bare string."""
    return kind.value if isinstance(kind, EventKind) else str(kind)


def _json_safe(value: Any) -> Any:
    """Convert ``value`` into something :func:`json.dumps` accepts."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=str)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_safe(getattr(value, item.name)) for item in fields(value)}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _json_safe(dump())
    return str(value)


def _task_label(task_id: str | None) -> str:
    """Render a task id as ``t3`` (or return ``""`` when there is none)."""
    if not task_id:
        return ""
    return task_id if task_id.startswith("t") else f"t{task_id}"


def _count(event: Event) -> int:
    """Return how many tasks ``event`` refers to, from ``data``."""
    count = event.data.get("count")
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    tasks = event.data.get("tasks")
    if isinstance(tasks, (list, tuple, set, frozenset)):
        return len(tasks)
    return 0


def _detail(event: Event) -> str:
    """Return the trailing detail of a rendered line, best effort."""
    files = event.data.get("files")
    if isinstance(files, int) and not isinstance(files, bool):
        return f"wrote {files} file(s)"
    if event.message:
        return event.message
    summary = event.data.get("summary")
    if isinstance(summary, str) and summary:
        return summary
    return ""
