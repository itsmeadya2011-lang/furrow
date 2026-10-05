"""FastAPI web UI for Furrow: a live view of a single orchestrator run.

The orchestrator reports progress through a *synchronous* ``on_event`` callback
while a WebSocket can only be written from an ``async`` context, so the two are
bridged with a bounded :class:`asyncio.Queue`:

* the callback only ever calls ``put_nowait`` -- it never blocks the agent loop;
* a full queue drops its *oldest* event, because the cycle markers and the final
  ``halt`` are worth more to a human than a stale burst of per-task lines;
* the streaming loop waits on the run task *and* on a ``websocket.receive()``
  task, so a browser that vanishes is noticed even while the agent is silent.

The event log is rendered twice from the same contract: once in Python
(:func:`format_event`, for the server log and for ``POST /api/start`` runs) and
once in JavaScript (``HANDLERS`` in :data:`INDEX_HTML`, for the browser).  The
browser only ever writes ``textContent``; payloads contain LLM-authored text, so
``innerHTML`` would be an XSS sink.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Mapping
from typing import Any, Callable, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator

__all__ = ["EVENT_NAMES", "StartRequest", "app", "format_event", "run"]

#: Every event name defined by the orchestrator contract, in emission order.
EVENT_NAMES: tuple[str, ...] = (
    "cycle_start",
    "plan",
    "task_start",
    "task_complete",
    "task_failed",
    "task_blocked",
    "tests",
    "cycle_end",
    "error",
    "halt",
)

#: Buffered events per connection before the oldest one is dropped.
_QUEUE_MAXSIZE = 1000

#: Default cap for a single rendered log line.
_TEXT_LIMIT = 240

app = FastAPI(title="Furrow")

#: Guards the `_current` reference only.  `GET` handlers may run on a different
#: event loop than the websocket that started the run, so a `threading.Lock` is
#: the portable choice: an `asyncio.Lock` created at import time binds to
#: whichever loop imported it first and raises when awaited from another loop.
_state_lock = threading.Lock()
_current: Optional["Orchestrator"] = None

#: Strong references to runs started over HTTP, so the event loop cannot garbage
#: collect a running task that nobody awaits.
_background_runs: set[asyncio.Task[None]] = set()

#: Websocket close code for a rejected (unusable) request.
_CLOSE_INVALID = 1008

logger = logging.getLogger(__name__)
event_logger = logging.getLogger("furrow.web.events")
_events_logger_ready = False


class _ClientGone(Exception):
    """Internal signal: the websocket is no longer writable."""


class StartRequest(BaseModel):
    """Validated body of ``POST /api/start``."""

    goal: str = Field(min_length=1, description="What Furrow should accomplish.")
    model: Optional[str] = Field(default=None, description="Optional model override.")

    @field_validator("goal")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        """Strip `value` and reject a goal that is empty after stripping.

        Args:
            value: The raw ``goal`` field from the request body.

        Returns:
            The stripped goal.

        Raises:
            ValueError: If the goal is blank.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("goal must not be blank")
        return stripped


def _text(value: Any, limit: int = _TEXT_LIMIT) -> str:
    """Collapse `value` onto one trimmed line of at most `limit` characters.

    Args:
        value: Any payload value, possibly `None`.
        limit: Maximum rendered length; longer values are ellipsised.

    Returns:
        A single-line string, empty when `value` is `None`.
    """
    if value is None:
        return ""
    flat = " ".join(str(value).split())
    if len(flat) <= limit:
        return flat
    return flat[: max(0, limit - 3)] + "..."


def _format_plan(data: Mapping[str, Any]) -> list[str]:
    """Render a ``plan`` event as a header line plus one line per planned task.

    The ``plan`` payload is passed through verbatim by the orchestrator, so it
    may be the model dump (an object with a ``tasks`` list) or the task list
    itself; both shapes are accepted.

    Args:
        data: The ``plan`` event payload (``cycle``, ``plan``, ``rationale``).

    Returns:
        At least one non-empty line.
    """
    rationale = _text(data.get("rationale"), 200) or "(no rationale)"
    lines = [f"plan · {rationale}"]
    plan = data.get("plan")
    tasks = plan.get("tasks") if isinstance(plan, Mapping) else plan
    if isinstance(tasks, list):
        if not tasks:
            lines.append("    (no tasks planned)")
        for task in tasks:
            if isinstance(task, Mapping):
                lines.append(f"    #{task.get('id')} {_text(task.get('description'), 160)}")
            else:
                lines.append(f"    {_text(task, 160)}")
    elif plan is not None:
        lines.append(f"    {_text(plan, 200)}")
    return lines


def _format_tests(data: Mapping[str, Any]) -> list[str]:
    """Render a ``tests`` event as a verdict line plus one line per failure.

    Args:
        data: The ``tests`` event payload.

    Returns:
        At least one non-empty line.
    """
    passed = bool(data.get("passed"))
    lines = [f"tests {'PASS' if passed else 'FAIL'} · {_text(data.get('summary'))}"]
    failures = data.get("failures")
    if isinstance(failures, list):
        lines.extend(f"    ! {_text(item, 200)}" for item in failures)
    return lines


#: Renderers keyed by event name; each returns at least one non-empty line.
#: This mirrors the browser-side ``HANDLERS`` map one for one.
_FORMATTERS: dict[str, Callable[[Mapping[str, Any]], list[str]]] = {
    "cycle_start": lambda d: [f"── cycle {d.get('cycle')} · {_text(d.get('goal'), 160)}"],
    "plan": _format_plan,
    "task_start": lambda d: [f"  → [{d.get('task_id')}] {_text(d.get('description'))}"],
    "task_complete": lambda d: [f"  ✔ [{d.get('task_id')}] {_text(d.get('result'))}"],
    "task_failed": lambda d: [f"  ✘ [{d.get('task_id')}] {_text(d.get('error'))}"],
    "task_blocked": lambda d: [
        f"  ⊘ [{d.get('task_id')}] blocked: {_text(d.get('reason'), 160)}"
    ],
    "tests": _format_tests,
    "cycle_end": lambda d: [
        f"── cycle {d.get('cycle')} end · done={d.get('done')}"
        f"{' · ' + _text(d.get('reason'), 160) if d.get('reason') else ''}"
    ],
    "error": lambda d: [f"!! {_text(d.get('message'), 300)}"],
    "halt": lambda d: [
        f"══ halt · {d.get('cycles')} cycle(s) · done={d.get('done')}"
        f" · {_text(d.get('reason'), 200)}"
    ],
}


def _is_degenerate(line: str) -> bool:
    """Return `True` when `line` carries no information at all.

    A payload that is missing every field still renders punctuation (``"!! "``);
    such a line is worse than useless in a log, so it is replaced by a visible
    placeholder instead.

    Args:
        line: A rendered log line.

    Returns:
        `True` when the line has no alphanumeric content.
    """
    return not any(char.isalnum() for char in line)


def format_event(name: str, data: Mapping[str, Any]) -> list[str]:
    """Render one orchestrator event as human-readable log lines.

    Args:
        name: An event name; normally one of :data:`EVENT_NAMES`.
        data: The event payload exactly as the orchestrator sent it.

    Returns:
        One or more non-empty lines.  An unknown event name, or a payload that
        renders to nothing, produces a visible placeholder line rather than
        silence, so drift between the orchestrator and this module shows up in
        the log instead of disappearing.
    """
    renderer = _FORMATTERS.get(name)
    if renderer is None:
        return [f"?? unhandled event: {name}"]
    lines = [line for line in renderer(data) if not _is_degenerate(line)]
    return lines or [f"?? empty payload for event: {name}"]


def _event_logger() -> logging.Logger:
    """Return the logger used for the event stream, attaching a handler once.

    uvicorn replaces the global logging configuration while starting up, which
    discards handlers installed at import time, so the handler is attached on
    first use instead.  ``propagate`` is disabled so the stream does not depend
    on the root logger being configured.

    Returns:
        The shared `furrow.web.events` logger.
    """
    global _events_logger_ready
    if not _events_logger_ready:
        _events_logger_ready = True
        if not event_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
            )
            event_logger.addHandler(handler)
            event_logger.setLevel(logging.INFO)
            event_logger.propagate = False
    return event_logger


def _log_event(name: str, payload: Mapping[str, Any]) -> None:
    """Write one event to the server log, flagging names outside the contract.

    Args:
        name: The event name.
        payload: The event payload.
    """
    log = _event_logger()
    level = logging.INFO if name in EVENT_NAMES else logging.WARNING
    for line in format_event(name, payload):
        log.log(level, "%s", line)


def make_callback(
    queue: Optional["asyncio.Queue[dict[str, Any]]"] = None,
) -> Callable[[str, dict[str, Any]], None]:
    """Build the synchronous ``on_event`` callback for a run.

    Args:
        queue: Destination queue, or `None` to log only (used by
            ``POST /api/start``, which has no socket to stream to).

    Returns:
        A callback that never raises and never blocks the caller.
    """

    def emit(name: str, payload: dict[str, Any]) -> None:
        try:
            _log_event(name, payload)
            if queue is not None:
                _offer(queue, {"event": name, "data": payload})
        except Exception:  # noqa: BLE001 - the agent must not fail because of logging
            logger.debug("dropping event %s", name, exc_info=True)

    return emit


def _offer(queue: "asyncio.Queue[dict[str, Any]]", item: dict[str, Any]) -> None:
    """Enqueue `item` without blocking and without raising.

    ``put_nowait`` keeps the synchronous callback from ever stalling the agent
    loop.  When the consumer is far behind, the oldest queued event is dropped
    rather than the newest, so the terminal state of the run survives a burst.

    Args:
        queue: The bounded bridge queue.
        item: The message to enqueue.
    """
    try:
        queue.put_nowait(item)
        return
    except asyncio.QueueFull:
        pass
    with contextlib.suppress(asyncio.QueueEmpty):
        queue.get_nowait()
    with contextlib.suppress(asyncio.QueueFull):
        queue.put_nowait(item)


async def _flush(websocket: WebSocket, queue: "asyncio.Queue[dict[str, Any]]") -> None:
    """Send every buffered event to `websocket`.

    Args:
        websocket: The connected socket.
        queue: The bridge queue to drain.

    Raises:
        _ClientGone: When a send fails, which always means the peer is gone.
    """
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            await websocket.send_json(item)
        except Exception as exc:  # noqa: BLE001 - any send failure is a dead socket
            raise _ClientGone(f"send failed: {exc}") from exc


async def _stream_run(
    websocket: WebSocket,
    task: "asyncio.Task[None]",
    queue: "asyncio.Queue[dict[str, Any]]",
) -> bool:
    """Relay queued events until the run ends or the client disappears.

    A second task awaits ``websocket.receive()`` so a disconnect is noticed even
    while the orchestrator is quiet.  The loop stops as soon as the run task is
    done, so events emitted during the very last step are not left behind.

    ``put_nowait`` cannot wake the relay (it must stay non-blocking), so while
    the queue is empty the relay waits on the run task, on a client frame, or on
    :data:`_FLUSH_INTERVAL` -- whichever comes first.  That bounds how long a
    freshly queued event can sit unsent while costing one wakeup per interval.

    Args:
        websocket: The connected socket.
        task: The task running ``Orchestrator.run()``.
        queue: The bridge queue filled by the orchestrator callback.

    Returns:
        ``True`` when the run finished on its own and the queue was fully
        drained, ``False`` when the client went away first.

    Raises:
        _ClientGone: When a send fails because the socket is dead.
    """
    reader = asyncio.create_task(websocket.receive())
    try:
        while not task.done():
            if queue.empty():
                done, _pending = await asyncio.wait(
                    {task, reader},
                    timeout=_FLUSH_INTERVAL,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if reader in done:
                    return False
            await _flush(websocket, queue)
    finally:
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await reader
    await _flush(websocket, queue)
    return True


async def _send_safe(websocket: WebSocket, message: dict[str, Any]) -> None:
    """Best-effort JSON send that never raises into the caller.

    Args:
        websocket: The connected socket.
        message: The message to send.
    """
    try:
        await websocket.send_json(message)
    except Exception:  # noqa: BLE001 - the peer may already be gone
        logger.debug("dropping %s: socket is gone", message.get("event"), exc_info=True)


async def _close_safe(websocket: WebSocket, code: int = 1000) -> None:
    """Close `websocket`, ignoring an already-closed socket.

    Args:
        websocket: The socket to close.
        code: Websocket close code to request.
    """
    with contextlib.suppress(Exception):
        await websocket.close(code=code)


async def _shutdown_task(task: Optional["asyncio.Task[None]"]) -> None:
    """Cancel `task` when it is still running and swallow its outcome.

    Args:
        task: The run task, or `None` when no run was started.
    """
    if task is None:
        return
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


def _task_failure(task: "asyncio.Task[None]") -> Optional[str]:
    """Return a one-line description of an unexpected run failure.

    Args:
        task: The finished run task.

    Returns:
        ``None`` when the task succeeded or was cancelled, otherwise a message.
    """
    if not task.done() or task.cancelled():
        return None
    exc = task.exception()
    if exc is None:
        return None
    return f"{type(exc).__name__}: {exc}"


def read_status(orchestrator: Optional[Orchestrator]) -> Optional[dict[str, Any]]:
    """Return a JSON-serialisable snapshot of `orchestrator`, or `None`.

    The ``status`` property is read defensively so that a change in its shape
    can never turn a `GET` handler into a 500.

    Args:
        orchestrator: The orchestrator to inspect, or `None`.

    Returns:
        The status mapping, or `None` when unavailable.
    """
    if orchestrator is None:
        return None
    try:
        status = orchestrator.status
    except Exception:  # noqa: BLE001 - never fail a GET on a status mismatch
        logger.exception("orchestrator status is unavailable")
        return None
    return status if isinstance(status, dict) else None


def set_current(orchestrator: Orchestrator) -> None:
    """Record `orchestrator` as the most recent run.

    Args:
        orchestrator: The orchestrator that is about to run.
    """
    global _current
    with _state_lock:
        _current = orchestrator


def get_current() -> Optional[Orchestrator]:
    """Return the most recent orchestrator, or `None` before the first run.

    Returns:
        The orchestrator started by the last websocket connection or HTTP call.
    """
    with _state_lock:
        return _current


def _settings_for(model: Optional[str]) -> Settings:
    """Build run-scoped settings, optionally overriding the model.

    A model override replaces ``model`` *and* the three per-role models, because
    the agents pass ``planner_model``/``worker_model``/``tester_model``
    explicitly to the LLM client, so overriding ``model`` alone would silently
    have no effect.  The global ``settings`` singleton is copied, never mutated.

    Args:
        model: The requested model name, or `None` to use the configured values.

    Returns:
        Settings for this run only.
    """
    if not model:
        return settings
    return settings.model_copy(
        update={
            "model": model,
            "planner_model": model,
            "worker_model": model,
            "tester_model": model,
        }
    )


def _closed_payload(orchestrator: Orchestrator) -> dict[str, Any]:
    """Build the payload of the final ``closed`` message.

    Args:
        orchestrator: The run that just finished.

    Returns:
        A mapping with ``running`` and, when available, the final ``status``.
    """
    status = read_status(orchestrator)
    if status is None:
        return {"running": False}
    return {"running": False, "status": status}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page UI as one self-contained document."""
    return HTMLResponse(content=INDEX_HTML)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Report process liveness."""
    return {"status": "ok"}


@app.get("/api/status")
async def status() -> JSONResponse:
    """Return the most recent run's ``Orchestrator.status`` snapshot.

    Returns:
        ``200`` with the status mapping, or ``404`` with
        ``{"running": false, "detail": "no run yet"}`` before the first run.
    """
    snapshot = read_status(get_current())
    if snapshot is None:
        return JSONResponse(status_code=404, content={"running": False, "detail": "no run yet"})
    return JSONResponse(content=snapshot)


@app.post("/api/start")
async def start_run(request: StartRequest) -> dict[str, Any]:
    """Start a background run from a JSON body and return immediately.

    The run logs every event to the server log and its state is readable from
    ``GET /api/status``; use the ``/ws`` endpoint to stream events to a browser.

    Args:
        request: The validated start request.

    Returns:
        ``{"started": True, "goal": <goal>}``.

    Raises:
        HTTPException: ``422`` for an invalid body, ``409`` while a run is live.
    """
    snapshot = read_status(get_current())
    if snapshot is not None and bool(snapshot.get("running")):
        raise HTTPException(status_code=409, detail="a run is already in progress")
    orchestrator = Orchestrator(
        goal=request.goal,
        on_event=make_callback(None),
        settings=_settings_for(request.model),
    )
    set_current(orchestrator)
    task = asyncio.create_task(orchestrator.run(), name="furrow-http-run")
    _background_runs.add(task)
    task.add_done_callback(_background_runs.discard)
    return {"started": True, "goal": request.goal}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Stream one orchestrator run to a browser as JSON event messages.

    The client sends ``{"goal": str, "model": str | None}``; the server then
    forwards every orchestrator event as ``{"event": name, "data": payload}``
    until the run halts, the client disconnects, or the socket fails.

    Args:
        websocket: The accepted websocket connection.
    """
    await websocket.accept()
    task: Optional["asyncio.Task[None]"] = None
    try:
        message = await websocket.receive_json()
        data: Mapping[str, Any] = message if isinstance(message, dict) else {}
        goal = str(data.get("goal") or "").strip()
        if not goal:
            await _send_safe(
                websocket,
                {"event": "error", "data": {"message": "a non-empty goal is required"}},
            )
            await _close_safe(websocket, _CLOSE_INVALID)
            return
        model = data.get("model")
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        orchestrator = Orchestrator(
            goal=goal,
            on_event=make_callback(queue),
            settings=_settings_for(str(model) if model else None),
        )
        set_current(orchestrator)
        task = asyncio.create_task(orchestrator.run(), name="furrow-ws-run")
        if await _stream_run(websocket, task, queue):
            failure = _task_failure(task)
            if failure is not None:
                await _send_safe(websocket, {"event": "error", "data": {"message": failure}})
            await _send_safe(websocket, {"event": "closed", "data": _closed_payload(orchestrator)})
            await _close_safe(websocket)
    except WebSocketDisconnect:
        logger.debug("websocket client disconnected")
    except _ClientGone as exc:
        logger.info("websocket closed by peer: %s", exc)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - a bad request must not kill the server
        logger.exception("websocket run failed")
        await _send_safe(
            websocket, {"event": "error", "data": {"message": "internal server error"}}
        )
        await _close_safe(websocket)
    finally:
        await _shutdown_task(task)


def run(host: str = "0.0.0.0", port: int = 8000, *, log_level: Optional[str] = None) -> None:
    """Serve the UI with uvicorn. Blocks until the server exits.

    Args:
        host: Interface to bind.  ``0.0.0.0`` publishes an unauthenticated
            agent that can edit files and spend API money; prefer ``127.0.0.1``.
        port: TCP port to listen on.
        log_level: Optional uvicorn log level, e.g. ``"debug"``.
    """
    level_name = (log_level or settings.log_level).upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO))
    uvicorn.run(app, host=host, port=port, log_level=log_level)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow</title>
<style>
  :root { color-scheme: dark; }
  body { background: #11151c; color: #d6deeb; margin: 0; padding: 1.5rem;
         font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  h1 { margin: 0 0 1rem; font-size: 1rem; letter-spacing: .12em;
       text-transform: uppercase; color: #7aa2f7; }
  form { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: .9rem; }
  input { flex: 1 1 16rem; padding: .5rem .6rem; font: inherit; color: inherit;
          background: #1a1f2b; border: 1px solid #2b3345; border-radius: 4px; }
  input#model { flex: 0 1 15rem; }
  button { padding: .5rem 1.1rem; font: inherit; color: #fff; cursor: pointer;
           background: #3d59a1; border: 0; border-radius: 4px; }
  button[disabled] { background: #39405a; cursor: not-allowed; opacity: .55; }
  #plan { margin: 0 0 .8rem 1rem; padding: 0; list-style: none; }
  #plan li { color: #bb9af7; }
  #log { height: 66vh; overflow-y: auto; padding: .75rem; background: #0b0e14;
         border: 1px solid #2b3345; border-radius: 6px;
         white-space: pre-wrap; word-break: break-word; }
  #status { margin: .7rem 0 0; color: #6b7794; }
  .cycle { color: #7aa2f7; font-weight: 600; }
  .plan  { color: #bb9af7; }
  .ok    { color: #9ece6a; }
  .fail  { color: #f7768e; }
  .warn  { color: #e0af68; }
  .done  { color: #7dcfff; }
  .info  { color: #6b7794; }
</style>
</head>
<body>
<h1>Furrow</h1>
<form id="form">
  <input id="goal" placeholder="What should Furrow build?" autocomplete="off" required />
  <input id="model" placeholder="model (optional, all roles)" autocomplete="off" />
  <button id="go" type="submit">Start</button>
</form>
<ul id="plan"></ul>
<div id="log"></div>
<p id="status">idle</p>
<script>
"use strict";

var form = document.getElementById("form");
var goalInput = document.getElementById("goal");
var modelInput = document.getElementById("model");
var goButton = document.getElementById("go");
var logBox = document.getElementById("log");
var planBox = document.getElementById("plan");
var statusBox = document.getElementById("status");
var CONNECT_TIMEOUT_MS = 8000;
var socket = null;

function socketUrl() {
  var scheme = location.protocol === "https:" ? "wss://" : "ws://";
  return scheme + location.host + "/ws";
}

function addLine(text, cls) {
  var line = document.createElement("div");
  line.className = cls ? cls : "info";
  // textContent only: payloads contain LLM-authored text and must never be
  // parsed as HTML.
  line.textContent = String(text);
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

function clearNode(node) {
  while (node.firstChild) { node.removeChild(node.firstChild); }
}

function setStatus(text) { statusBox.textContent = text; }

function unlock() { goButton.disabled = false; }

function finish(reason) {
  unlock();
  setStatus("finished · " + reason);
}

function short(value, limit) {
  if (value === null || value === undefined) { return ""; }
  var flat = String(value).replace(/\\s+/g, " ").trim();
  if (flat.length <= limit) { return flat; }
  return flat.slice(0, limit - 3) + "...";
}

function taskList(tasks) {
  for (var i = 0; i < tasks.length; i += 1) {
    var task = tasks[i];
    var label = (task !== null && typeof task === "object")
      ? "#" + task.id + " " + short(task.description, 140)
      : short(task, 140);
    var item = document.createElement("li");
    item.textContent = label;
    planBox.appendChild(item);
  }
}

// One entry per event in the orchestrator contract; mirrors
// furrow.web.server._FORMATTERS.
var HANDLERS = {
  cycle_start: function (d) {
    addLine("── cycle " + d.cycle + " · " + short(d.goal, 160), "cycle");
  },
  plan: function (d) {
    addLine("plan · " + (d.rationale ? short(d.rationale, 200) : "(no rationale)"), "plan");
    var plan = d.plan;
    var tasks = (Array.isArray(plan)) ? plan
      : (plan !== null && typeof plan === "object" ? plan.tasks : null);
    if (Array.isArray(tasks)) {
      if (!tasks.length) { addLine("  (no tasks planned)", "plan"); }
      taskList(tasks);
    } else if (plan !== null && plan !== undefined) {
      addLine("  " + short(plan, 200), "plan");
    }
  },
  task_start: function (d) {
    addLine("  → [" + d.task_id + "] " + short(d.description, 200), "info");
  },
  task_complete: function (d) {
    addLine("  ✔ [" + d.task_id + "] " + short(d.result, 240), "ok");
  },
  task_failed: function (d) {
    addLine("  ✘ [" + d.task_id + "] " + short(d.error, 240), "fail");
  },
  task_blocked: function (d) {
    addLine("  ⊘ [" + d.task_id + "] blocked: " + short(d.reason, 160), "warn");
  },
  tests: function (d) {
    addLine("tests " + (d.passed ? "PASS" : "FAIL") + " · " + short(d.summary, 240),
            d.passed ? "ok" : "fail");
    var failures = Array.isArray(d.failures) ? d.failures : [];
    for (var i = 0; i < failures.length; i += 1) {
      addLine("    ! " + short(failures[i], 200), "fail");
    }
  },
  cycle_end: function (d) {
    var reason = d.reason ? " · " + short(d.reason, 160) : "";
    addLine("── cycle " + d.cycle + " end · done=" + d.done + reason, "cycle");
  },
  error: function (d) {
    addLine("!! " + short(d.message, 300), "fail");
  },
  halt: function (d) {
    addLine("══ halt · " + d.cycles + " cycle(s) · done=" + d.done
            + " · " + short(d.reason, 200), "done");
    finish("halted after " + d.cycles + " cycle(s)");
  },
  closed: function (d) {
    if (d && d.status && d.status.halt_reason) {
      addLine("· " + short(d.status.halt_reason, 200), "info");
    }
    finish("connection closed");
  }
};

function handleMessage(event) {
  var frame;
  try {
    frame = JSON.parse(event.data);
  } catch (err) {
    addLine("?? unparseable frame from server", "warn");
    return;
  }
  var name = (frame && frame.event) ? String(frame.event) : "";
  var data = (frame && frame.data && typeof frame.data === "object") ? frame.data : {};
  if (!Object.prototype.hasOwnProperty.call(HANDLERS, name)) {
    addLine("?? unhandled event: " + (name || "(none)"), "warn");
    return;
  }
  try {
    HANDLERS[name](data);
  } catch (err) {
    addLine("?? handler failed for " + name, "warn");
  }
}

function stop() {
  if (socket) {
    socket.close();
    socket = null;
  }
  unlock();
}

form.addEventListener("submit", function (event) {
  event.preventDefault();
  var goal = goalInput.value.trim();
  if (!goal) {
    addLine("!! a goal is required", "fail");
    return;
  }
  if (socket) { return; }
  clearNode(logBox);
  clearNode(planBox);
  goButton.disabled = true;
  setStatus("starting…");

  var url = socketUrl();
  var opened = false;
  var timer = setTimeout(function () {
    if (!opened) {
      setStatus("still connecting…");
      addLine("?? no websocket yet at " + url, "warn");
    }
  }, CONNECT_TIMEOUT_MS);

  socket = new WebSocket(url);
  socket.onopen = function () {
    opened = true;
    clearTimeout(timer);
    setStatus("running");
    addLine("· connected to " + url, "info");
  };
  socket.onmessage = handleMessage;
  socket.onerror = function () {
    addLine("!! websocket error", "fail");
    setStatus("connection error");
  };
  socket.onclose = function (event) {
    clearTimeout(timer);
    socket = null;
    unlock();
    var code = event && typeof event.code === "number" ? event.code : "?";
    addLine("· socket closed (code " + code + ")", "info");
    if (!opened) { setStatus("connection failed"); }
  };
  socket.send(JSON.stringify({ goal: goal, model: modelInput.value.trim() || null }));
});
</script>
</body>
</html>
"""
