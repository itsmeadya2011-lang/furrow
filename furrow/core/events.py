"""Progress events emitted by the orchestrator and consumed by the web/CLI front-ends.

This module is deliberately dependency-light: it depends only on the standard library,
pydantic (already a core dependency) and rich (already a core dependency). It knows
nothing about planners, workers or the orchestration loop, so the web layer can import it
without pulling in the LLM client or provider SDKs.

Design notes
------------
* **JSON everywhere.** Every event dict is a plain ``dict[str, Any]`` and every value is
  run through :func:`_jsonable` before it is stored or handed to a subscriber. The web layer
  serialises events straight to WebSocket JSON frames, so ``json.dumps(event)`` must always
  succeed -- a ``Path`` or a pydantic model in a field must never reach the wire raw.
* **Cycle auto-fill.** :meth:`EventEmitter.emit` stamps a ``"cycle"`` key from
  ``self.current_cycle`` (an int initialised to ``0``) when the caller does not pass one
  explicitly, so callers never have to thread the cycle number through every call site.
* **Defensive rendering.** Mirroring an event to the rich console can never abort a run.
  All dynamic text is escaped with :func:`rich.markup.escape`: an LLM result containing
  ``[bold]`` would otherwise raise ``MarkupError`` and kill the loop, which is why this
  helper exists alongside a plain ``console.print``.
* **Defensive subscribers.** A listener that raises is reported via the console and then
  dropped; it never propagates into the orchestrator.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.markup import escape

__all__ = ["Event", "EventEmitter", "EventType"]


class EventType(str, Enum):
    """Kinds of progress event the orchestrator may emit.

    The value of each member is the wire string used as the event's ``"type"`` field, and
    code must read ``EventType.LOG.value`` -- never ``str(EventType.LOG)``. This is a real
    footgun rather than a style rule: for a ``str``/``Enum`` mixin the ``__str__`` behaviour
    is version-dependent. On Python 3.11+ ``str(EventType.LOG)`` returns ``"log"`` (the
    mixin value), but on 3.10 it returns ``"EventType.LOG"`` (the ``Enum.__str__`` repr),
    which is not a valid wire value and silently breaks every ``== EventType.X.value``
    comparison downstream (e.g. :func:`_render_event`'s dispatch). Python 3.13 additionally
    changes ``__format__`` again. ``.value`` is stable across all of them, which is why the
    emitter routes the type through :func:`_jsonable`, the single place that maps
    ``Enum -> .value``.
    """

    #: Emitted once, before the first cycle. Carries goal, workspace, provider, models.
    RUN_START = "run_start"
    #: Start of a planner cycle. Carries ``cycle``.
    CYCLE_START = "cycle_start"
    #: A plan was produced. Carries ``rationale``, ``task_count`` and ``tasks``.
    PLAN = "plan"
    #: A worker started a task. Carries ``task_id``, ``description``, ``files``,
    #: ``dependencies``.
    TASK_START = "task_start"
    #: A worker finished a task. Carries ``task_id``, ``description``, ``status``,
    #: ``written_files`` and ``result``.
    TASK_END = "task_end"
    #: The tester ran. Carries ``passed``, ``skipped``, ``summary``, ``failures``.
    TEST = "test"
    #: End of a cycle. Carries ``completed``, ``failed``, ``skipped``, ``passed``.
    CYCLE_END = "cycle_end"
    #: Emitted once, after the loop exits. Carries ``cycles``, totals and ``stopped_because``.
    RUN_END = "run_end"
    #: Free-form diagnostic line. Carries ``message`` (a plain string).
    LOG = "log"
    #: An error that did not abort the run but was noteworthy. Carries ``stage`` and
    #: ``message``; the ``message`` is always a JSON-safe string.
    ERROR = "error"


#: A JSON-serialisable event dict. Always contains a ``"type"`` key (the
#: :class:`EventType` value as a string) and always contains a ``"cycle"`` key
#: (auto-stamped by :meth:`EventEmitter.emit` unless the caller passed one).
Event = dict[str, Any]


def _jsonable(value: Any) -> Any:
    """Coerce ``value`` into something ``json.dumps`` can encode.

    Handles the types the orchestrator and front-ends actually emit: ``Path`` -> ``str``,
    :class:`Enum` -> ``.value``, pydantic :class:`~pydantic.BaseModel` ->
    ``model_dump(mode="json")``, plus dict/list/tuple/set recursion. Anything else falls
    back to ``str(value)`` -- the web layer treats an unknown shape as a string rather
    than dropping the event or crashing the run.

    The ``Enum`` arm must come *before* the scalar ``isinstance`` arm: every member of a
    ``str``/``int``/``float`` mixin enum such as :class:`EventType` or
    :class:`~furrow.models.TaskStatus` *is* an instance of its mixin type, so a plain
    ``isinstance(value, str)`` fast path would hand the enum member straight back -- and
    ``str()``-ing that on Python 3.10 yields ``"EventType.LOG"`` rather than ``"log"``.
    """
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


class EventEmitter:
    """Minimal, dependency-light progress-event bus.

    An emitter owns an optional rich :class:`~rich.console.Console` (for human-readable
    mirroring), an append-only :attr:`history` of every event, and a list of subscriber
    callbacks. The web layer registers subscribers to receive events and forward them as
    WebSocket JSON frames; the CLI may pass a console for live feedback.

    Args:
        on_event: Optional callable invoked for every emitted event, after the console
            mirror and history append. Useful for the web layer to push events over a
            socket.
        console: Optional rich console. When provided, each event is mirrored (escaped) to
            it; when ``None`` no rendering happens and the run still works, driven purely
            by subscribers / history.
        keep_history: When ``True`` (default) every event is retained in :attr:`history`
            for later inspection.
    """

    def __init__(
        self,
        on_event: Callable[[Event], None] | None = None,
        *,
        console: Console | None = None,
        keep_history: bool = True,
    ) -> None:
        """See class docstring."""
        self._console: Console | None = console
        self._subscribers: list[Callable[[Event], None]] = []
        if on_event is not None:
            self._subscribers.append(on_event)
        self._history: list[Event] = []
        self._keep_history: bool = keep_history
        #: Current cycle number, stamped onto events that don't pass one. Initialised to
        #: ``0`` so that ``RUN_START`` (emitted before any cycle) lands on cycle ``0``.
        self.current_cycle: int = 0
        if self._console is not None:
            self._subscribers.append(self._console_render)

    @property
    def history(self) -> list[Event]:
        """All events seen since the last :meth:`clear`, in emission order."""
        return self._history

    def clear(self) -> None:
        """Drop the retained event history (subscribers are unaffected)."""
        self._history.clear()

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """Register ``callback`` to receive every emitted event.

        Returns:
            A zero-argument callable that removes ``callback`` from the emitter. Calling
            it more than once is a no-op.
        """
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            """Remove the original callback; idempotent."""
            with suppress(ValueError):
                self._subscribers.remove(callback)

        return unsubscribe

    def emit(self, event_type: EventType | str, /, **fields: Any) -> Event:
        """Build one event dict, mirror it to the console, store it and fan it out.

        The event is ``{"type": <wire value>, **fields}`` -- ``EventType`` members become
        their ``.value`` and a plain ``str`` is preserved verbatim, both via
        :func:`_jsonable`. A ``"cycle"`` key is auto-filled from :attr:`current_cycle` when
        the caller did not supply one, so cycle context never has to be threaded through
        every call site. Every field value is passed through :func:`_jsonable` before
        storage so the result is always JSON serialisable.

        Failures are quarantined: a crashing console render or a raising subscriber is
        reported and swallowed, never propagated into the orchestrator.
        """
        event: Event = {"type": _jsonable(event_type)}
        fields.setdefault("cycle", self.current_cycle)
        event.update({key: _jsonable(val) for key, val in fields.items()})
        if self._keep_history:
            self._history.append(event)
        for callback in list(self._subscribers):
            try:
                callback(event)
            except Exception as exc:  # noqa: BLE001 - listeners must never propagate
                self._report_subscriber_error(exc)
        return event

    def _console_render(self, event: Event) -> None:
        """Mirror ``event`` to the rich console as a single escaped line.

        Rendering failures are swallowed so a broken console (e.g. a closed pipe during
        shutdown) cannot abort an in-flight run.
        """
        if self._console is None:
            return
        try:
            self._console.print(_render_event(event))
        except Exception:  # noqa: BLE001 - console mirrors must never break the loop
            return

    def _report_subscriber_error(self, exc: BaseException) -> None:
        """Print a one-line warning when a subscriber raises, without re-raising."""
        if self._console is None:
            return
        try:
            self._console.print(
                f"[red]furrow: event subscriber raised {type(exc).__name__}[/red]"
            )
        except Exception:  # noqa: BLE001 - reporting must never raise either
            return


def _render_event(event: Event) -> str:
    """Render ``event`` as a single, fully-escaped human-readable line.

    Every dynamic segment is passed through :func:`rich.markup.escape` so that an LLM's
    prose -- which routinely contains ``[bold]`` or other markup that rich would otherwise
    try to parse and raise :class:`rich.markup.MarkupError` for -- is printed verbatim.
    The escaped string contains no markup of its own, so ``console.print`` cannot raise.
    """
    etype = str(event.get("type", ""))
    cycle = event.get("cycle")
    header = f"(cycle {cycle}) " if cycle is not None else ""
    if etype == EventType.LOG.value:
        body = str(event.get("message", ""))
    elif etype == EventType.ERROR.value:
        body = f"error [{event.get('stage', '?')}]: {event.get('message', '')}"
    elif etype == EventType.TASK_START.value:
        body = f"start {event.get('task_id', '?')}: {event.get('description', '')}"
    elif etype == EventType.TASK_END.value:
        body = (
            f"done {event.get('task_id', '?')} -> {event.get('status', '?')} "
            f"({len(event.get('written_files', []) or [])} writes)"
        )
    elif etype == EventType.PLAN.value:
        body = (
            f"plan {event.get('task_count', 0)} tasks: "
            f"{event.get('rationale', '')}"
        )
    elif etype == EventType.TEST.value:
        body = (
            f"test passed={event.get('passed')} skipped={event.get('skipped')}: "
            f"{event.get('summary', '')}"
        )
    elif etype == EventType.CYCLE_END.value:
        body = (
            f"cycle end {event.get('completed')} ok / "
            f"{event.get('failed')} fail / {event.get('skipped')} skip"
        )
    elif etype == EventType.RUN_END.value:
        body = (
            f"run end: {event.get('stopped_because')} after "
            f"{event.get('cycles')} cycles"
        )
    elif etype == EventType.RUN_START.value:
        body = f"goal: {event.get('goal', '')}"
    elif etype == EventType.CYCLE_START.value:
        body = "cycle start"
    else:
        body = etype
    return escape(f"{header}{etype}: {body}")
