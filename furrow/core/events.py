"""Structured progress events for the run loop.

Furrow prints progress with ``rich`` straight to stdout, which serves the human
watching a terminal and serves nobody else: a browser cannot read painted text, and a
test cannot assert on it. This module is the seam between the loop and those other
consumers. The orchestrator emits :class:`Event` objects; subscribers - a rich renderer,
the web server's WebSocket pump, a test - decide what to do with them.

Deliberately stdlib-only. The orchestrator, the CLI, the web app and the test suite all
import this module, so depending on ``furrow.config`` / ``furrow.llm`` / ``furrow.models``
would make every one of them a candidate for an import cycle. Progress reporting is the
last thing that should be able to stop the run from working.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = ["Event", "EventEmitter", "EventType", "event_to_dict"]

# How deep `_jsonable` will walk a structure before giving up and emitting `repr`.
# Event payloads carry data parsed out of language-model output, which is untrusted: a
# model can hand back something self-referential or absurdly nested, and a recursive
# walk would either recurse until the interpreter raises RecursionError (killing the
# emitter that is merely trying to report progress) or expand into an unbounded amount
# of work on the socket pump. Twelve levels is far deeper than any real event payload
# and keeps the fallback cheap.
_MAX_DEPTH = 12


class EventType(str, Enum):
    """The closed vocabulary of things a consumer may want to know about.

    A fixed enum rather than free-form strings so that a consumer can switch
    exhaustively on what arrives, and so a typo at a call site is an ``AttributeError``
    immediately instead of a frame the web UI quietly drops forever.

    Subclasses ``str`` because the values cross a JSON boundary. That subclassing is
    necessary but emphatically not sufficient: on Python 3.10 ``str(EventType.LOG)``
    still yields ``"EventType.LOG"``, which is why :func:`event_to_dict` extracts
    ``.value`` itself instead of trusting ``str()``.
    """

    CYCLE = "cycle"
    PLAN = "plan"
    TASK_START = "task_start"
    TASK_DONE = "task_done"
    TEST = "test"
    LOG = "log"
    ERROR = "error"
    DONE = "done"


@dataclass
class Event:
    """One thing that happened, in a shape every consumer can read.

    ``data`` defaults to a fresh dict per instance via ``default_factory``; a plain
    ``{}`` default would be evaluated once at class creation and every event would share
    it, so a subscriber mutating one event's payload would corrupt every later event.
    ``ts`` is wall-clock because consumers care about when a cycle happened, not about
    ordering that the deque already encodes.
    """

    type: EventType
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class EventEmitter:
    """Fan-out point for :class:`Event` objects: bounded history plus subscribers.

    Both halves matter. History exists so a test, or a browser that connects mid-run, can
    see what already happened. Subscribers exist so a renderer does not have to poll.
    The two failure modes are handled deliberately: a bounded deque, because the loop
    runs forever and an unbounded list is a memory leak whose symptoms appear hours
    later, and per-subscriber exception isolation, because observability must never be
    able to fail the work it observes.
    """

    def __init__(
        self,
        callback: Callable[[Event], None] | None = None,
        *,
        history_limit: int = 1000,
    ) -> None:
        """Seed the emitter with an optional first subscriber and a history bound.

        ``history_limit <= 0`` is rejected rather than honoured. ``deque(maxlen=0)``
        accepts an event and silently discards it, so an emitter built that way looks
        wired up and records nothing - a confusing failure to debug from a symptom
        ("the web UI shows no past events") that has nothing to do with the real cause.
        A caller that genuinely wants no history can simply not use the history.
        """
        if history_limit <= 0:
            raise ValueError(f"history_limit must be positive, got {history_limit}")

        self._history: deque[Event] = deque(maxlen=history_limit)
        self._subscribers: list[Callable[[Event], None]] = []
        if callback is not None:
            self._subscribers.append(callback)

    def emit(self, type: EventType, message: str, **data: Any) -> None:
        """Record one event and hand it to every subscriber.

        Never raises. A subscriber here is a rich renderer or a WebSocket pump writing
        to a socket whose client may have vanished; that is an ordinary event of a
        long-running server, not an error in the run. Letting it escape would take down
        the orchestrator's cycle loop - the failure would be an observer's bug
        terminating the actual work, and observability must never be able to fail the
        work it observes. Each callback is therefore isolated, and an early subscriber
        that raises does not starve the ones registered after it.
        """
        event = Event(type=type, message=message, data=data)
        self._history.append(event)
        # Snapshot the list: a callback is allowed to call `on()` (or to emit again),
        # and mutating a list while iterating it would skip a subscriber entirely.
        for callback in list(self._subscribers):
            try:
                callback(event)
            except Exception:
                continue

    def on(self, callback: Callable[[Event], None]) -> None:
        """Add a subscriber, keeping registration order.

        Order is append order rather than a set or dict keyed on the callback, because
        rendering order should be reproducible: tests assert on the sequence of
        notifications, and two renderers that ran in a different order between runs
        would make for maddening, non-local failures. Duplicate registration is
        therefore also allowed, and delivers twice.
        """
        self._subscribers.append(callback)

    def of_type(self, *types: EventType) -> list[Event]:
        """Return recorded events of the given types, oldest first.

        Filtering in the emitter rather than at each call site keeps every consumer
        asking the same question the same way, and returning ``[]`` for a type that was
        never emitted is the normal "nothing matched" answer rather than an error: a
        browser polling for TEST events before the first test runs is not a bug.
        """
        wanted = set(types)
        return [event for event in self._history if event.type in wanted]

    @property
    def events(self) -> list[Event]:
        """A snapshot of the recorded history, oldest first.

        A copy, not the deque itself, so a caller that sorts or truncates the result
        cannot silently rewrite the emitter's history or trip over its size limit.
        """
        return list(self._history)

    def clear(self) -> None:
        """Drop the recorded history, keeping subscribers registered.

        Callers use this to bound memory in long runs and to give a test a clean slate;
        unsubscribing is a separate concern, so a clear must not silently stop
        delivery.
        """
        self._history.clear()


def event_to_dict(event: Event) -> dict[str, Any]:
    """Render one event as a JSON-serialisable dict for the WebSocket payload.

    The shape is fixed - ``{"type", "message", "data", "ts"}`` - so the browser has one
    contract to parse. Conversion of ``data`` is delegated to :func:`_jsonable` rather
    than to ``dataclasses.asdict`` or ``json.dumps(..., default=str)`` because event
    payloads mix model output, paths, datetimes and sets, and each of those needs a
    different, lossless-enough representation.
    """
    return {
        "type": _jsonable(event.type),
        "message": _jsonable(event.message),
        "data": _jsonable(event.data),
        "ts": _jsonable(event.ts),
    }


def _jsonable(obj: Any, depth: int = 0) -> Any:
    """Convert ``obj`` into something ``json.dumps`` accepts, without ever raising.

    The order of the arms below is load-bearing, not cosmetic. See the Enum comment
    inside for the sharpest edge; the rest is "most specific shape first, scalars last"
    so that a richer type is never swallowed by the branch above it.

    ``depth`` bounds the walk for untrusted, language-model-derived data: past
    ``_MAX_DEPTH`` levels the value degrades to ``repr`` rather than recursing further
    into a structure that may be self-referential.
    """
    if depth > _MAX_DEPTH:
        return repr(obj)

    # Enum FIRST, and this is the subtlest bug this module could have.
    # On Python 3.10 `class EventType(str, Enum)` does NOT give you a str whose text is
    # the value: `__str__` is inherited from `Enum`, not from `str`, so
    # `str(EventType.LOG)` == "EventType.LOG", not "log". If the scalar/str arm below
    # came first, every event would serialise its type as the string "EventType.LOG"
    # and the web UI would silently render nothing meaningful. `.value` is the only
    # place the actual wire token lives.
    if isinstance(obj, Enum):
        return obj.value

    # Duck-typed rather than `isinstance(obj, BaseModel)`: this module is stdlib-only,
    # and every pydantic v2 model exposes `model_dump`. Anything else answering to that
    # name is expected to behave the same way.
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            # `model_dump` leaves datetimes, Paths and nested models in place, so the
            # result still has to come back through here.
            return _jsonable(model_dump(), depth + 1)
        except RecursionError:
            return repr(obj)

    # `is_dataclass` is also true for the class itself, and a type is not an instance of
    # the payload shape we want to expand.
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return _jsonable(dataclasses.asdict(obj), depth + 1)
        except RecursionError:
            # A dataclass that refers to itself defeats `asdict`, which recurses without
            # a depth bound. Re-walking the fields by hand does terminate, because the
            # guard at the top of this function does.
            return {
                f.name: _jsonable(getattr(obj, f.name), depth + 1)
                for f in dataclasses.fields(obj)
            }

    if isinstance(obj, Path):
        return str(obj)

    # `datetime` is a subclass of `date`, so this one test covers both.
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, (set, frozenset)):
        # Members are converted before sorting so that e.g. a set of enums comes out as
        # plain strings; sorting before conversion would sort heterogeneous members and
        # raise TypeError on data the loop is only trying to report.
        members = [_jsonable(item, depth + 1) for item in obj]
        return sorted(members) if all(isinstance(item, str) for item in members) else members

    if isinstance(obj, bytes):
        # `errors="replace"` because a truncated byte stream from a killed subprocess is
        # expected, and raising here would lose the very diagnostic it describes.
        return obj.decode("utf-8", errors="replace")

    if isinstance(obj, Mapping):
        return {key: _jsonable(value, depth + 1) for key, value in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_jsonable(item, depth + 1) for item in obj]

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # `repr` rather than `str` for anything unrecognised: it is unambiguous about being
    # a fallback, so a consumer reading the payload can tell the object was not
    # understood rather than mistaking a pretty-printed object for real data.
    return repr(obj)
