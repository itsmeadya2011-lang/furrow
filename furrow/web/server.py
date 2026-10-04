"""Furrow's monitoring UI: a progress-streaming websocket plus a small REST surface.

This module is a *monitor*, not a product, so it deliberately keeps no persistent state:
runs live in an in-memory registry that is capped and dropped oldest-first.

The websocket contract
----------------------
``WS /ws`` accepts exactly one start message -- ``{"goal": "...", "model": "..."}`` (only
``goal`` is required) -- and then forwards every orchestrator progress event as

``{"type": "event", "event": <name>, "data": {...}, "ts": <float>}``

where ``<name>`` is one of the eight events emitted by
:mod:`furrow.core.orchestrator` with these payload keys:

=====================  ======================================================
event                  payload keys
=====================  ======================================================
``cycle_start``        ``cycle``, ``focus``
``plan``               ``cycle``, ``tasks``, ``rationale``
``task_start``         ``cycle``, ``task``
``task_done``          ``cycle``, ``task``, ``status``, ``skipped``, ``result``
``test``               ``cycle``, ``passed``, ``verified``, ``summary``, ``failures``
``cycle_end``          ``cycle``, ``changed_files``, ``stalled_cycles``
``halt``               ``reason``, ``cycles``, ``changed_files``, ``complete``
``planner_error``      ``cycle``, ``error``, ``consecutive``
=====================  ======================================================

Exactly one terminal frame follows: ``{"type": "result", "data": {...}}`` when the run
finished for any reason (``stopped_reason``, ``cycles``, ``changed_files``, ``tasks``) or
``{"type": "error", "data": {"message": ...}}`` when the run could not be started or
raised. Bad input -- malformed JSON, a blank or non-string ``goal`` -- is answered with an
``error`` frame and a clean close, never with an unhandled server error. A client
disconnect, a browser refresh or a ``{"type": "stop"}`` message cancels the in-flight run
task instead of leaving it spending tokens.

The REST contract
-----------------
``POST /api/start`` registers and launches a run (optionally overriding the model),
``GET /api/runs`` lists the retained runs and ``GET /api/runs/{run_id}`` returns one run
with its task list and recorded event trail.

Security: there is no authentication, CORS or rate limiting here, and every run spends
money on LLM calls, so :func:`run` binds ``127.0.0.1`` unless told otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)

#: Sent when the start message was not JSON (RFC 6455: unacceptable data).
WS_UNSUPPORTED_DATA = 1003
#: Sent when the start message decoded but carried no usable goal.
WS_POLICY_VIOLATION = 1008
#: Message telling a live stream to stop and be cancelled.
STOP_COMMAND = "stop"
#: Events retained per run for ``GET /api/runs/{run_id}``; the oldest are dropped.
MAX_RECORDED_EVENTS = 200
#: Runs retained in memory; the oldest is dropped once the registry is full.
MAX_TRACKED_RUNS = 50
#: ``Settings`` fields naming the per-role model, all overridden together by ``model``.
ROLE_MODEL_FIELDS = ("planner_model", "worker_model", "tester_model")
#: Reported when a run ends because its client went away.
CANCELLED_REASON = "cancelled: the client left"
#: Reported when the goal could not be read out of the start message.
INVALID_GOAL_MESSAGE = "goal must be a non-empty string"

app = FastAPI(title="Furrow")

#: Every retained run, oldest first. Process-local and intentionally unbounded in age.
_RUNS: dict[str, RunRecord] = {}
#: Monotonic run id source, so ids stay readable in the UI and in tests.
_RUN_IDS = itertools.count(1)


class StartRequest(BaseModel):
    """Body of ``POST /api/start``.

    ``model`` overrides the default model *and* every per-role model for that run only;
    the shared settings singleton is never mutated.
    """

    goal: str
    model: str | None = None

    @field_validator("goal")
    @classmethod
    def _goal_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("goal must not be empty")
        return value


class FrameSink(Protocol):
    """Where a run's frames go: a websocket queue, or nowhere at all."""

    def put_nowait(self, frame: dict[str, Any] | None) -> None:
        """Accept one frame, or ``None`` as the end-of-stream sentinel."""


class NullSink:
    """Frame sink for a run with no websocket attached (started over REST).

    The run registry is then the only observer, so frames are dropped rather than queued
    for a reader that will never arrive.
    """

    def put_nowait(self, frame: dict[str, Any] | None) -> None:
        """Discard the frame."""


@dataclass
class RunRecord:
    """One orchestrated run: its identity, its live progress and its final outcome."""

    id: str
    goal: str
    model: str | None
    started_at: float
    finished_at: float | None = None
    stopped_reason: str | None = None
    cycles: int = 0
    changed_files: list[str] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    runner: asyncio.Task[None] | None = None

    def summary(self) -> dict[str, Any]:
        """The listing shape: identity, outcome and progress counters."""
        return {
            "id": self.id,
            "goal": self.goal,
            "model": self.model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "running": self.runner is not None and not self.runner.done(),
            "stopped_reason": self.stopped_reason,
            "cycles": self.cycles,
            "changed_files": list(self.changed_files),
            "error": self.error,
            "event_count": len(self.events),
        }

    def result_payload(self) -> dict[str, Any]:
        """The terminal frame payload: the summary plus the finished task list."""
        return {**self.summary(), "tasks": [dict(task) for task in self.tasks]}

    def detail(self) -> dict[str, Any]:
        """Everything about the run, including its recorded event trail."""
        return {**self.result_payload(), "events": list(self.events)}


def register_run(goal: str, model: str | None = None) -> RunRecord:
    """Add a run to the registry, dropping the oldest entry once it is full."""
    record = RunRecord(id=f"run-{next(_RUN_IDS)}", goal=goal, model=model, started_at=time.time())
    _RUNS[record.id] = record
    while len(_RUNS) > MAX_TRACKED_RUNS:
        del _RUNS[next(iter(_RUNS))]
    return record


def event_frame(name: str, data: dict[str, Any]) -> dict[str, Any]:
    """Wrap one orchestrator progress event in the streaming envelope."""
    return {"type": "event", "event": name, "data": data, "ts": time.time()}


def frame(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """Wrap one terminal frame (``result`` or ``error``) in the streaming envelope."""
    return {"type": kind, "data": data, "ts": time.time()}


def error_frame(message: str) -> dict[str, Any]:
    """The terminal frame for a run that could not start or blew up."""
    return frame("error", {"message": message})


def parse_start(payload: object) -> tuple[str | None, str | None]:
    """Split a decoded start message into ``(goal, model)``.

    ``goal`` is ``None`` when the message cannot be used, which is the single condition
    the callers turn into an ``error`` frame. A non-string ``goal`` is rejected rather
    than coerced, so a caller never starts a run against the string ``"None"``.
    """
    if not isinstance(payload, dict):
        return None, None
    goal = payload.get("goal")
    model = payload.get("model")
    if not isinstance(goal, str) or not goal.strip():
        return None, None
    return goal, model if isinstance(model, str) and model.strip() else None


def client_for(model: str | None) -> LLMClient | None:
    """Build an isolated client for one run when ``model`` overrides the default.

    Returns ``None`` -- the orchestrator's own client, bound to the shared settings
    singleton -- when no override was asked for, so a plain run is completely unaffected.
    """
    if not model:
        return None
    update: dict[str, Any] = {"model": model}
    for name in ROLE_MODEL_FIELDS:
        update[name] = model
    return LLMClient(settings=settings.model_copy(update=update))


def build_orchestrator(
    goal: str,
    on_event: Callable[[str, dict[str, Any]], None],
    model: str | None = None,
) -> Orchestrator:
    """Create the engine for one run, wired to its progress hook."""
    return Orchestrator(goal=goal, client=client_for(model), on_event=on_event)


def event_hook(record: RunRecord, sink: FrameSink) -> Callable[[str, dict[str, Any]], None]:
    """Build the orchestrator's synchronous progress hook for one run.

    The hook is called from the loop that drives the orchestrator, so frames go straight
    into the queue without a thread hand-off. Every step is guarded: a hook that raised
    would take down the very run it is reporting on.
    """

    def hook(name: str, payload: dict[str, Any]) -> None:
        try:
            apply_event(record, name, payload)
            sink.put_nowait(event_frame(name, payload))
        except Exception:  # progress reporting must never kill the run it reports on
            logger.debug("could not forward event %s", name, exc_info=True)

    return hook


def apply_event(record: RunRecord, name: str, payload: dict[str, Any]) -> None:
    """Fold one progress event into the registry entry, so a run is queryable while live."""
    record.events.append({"event": name, "data": payload})
    del record.events[:-MAX_RECORDED_EVENTS]
    if name == "cycle_start":
        record.cycles = max(record.cycles, as_int(payload.get("cycle")))
    elif name == "cycle_end":
        merge_paths(record.changed_files, payload.get("changed_files"))
    elif name == "halt":
        record.cycles = as_int(payload.get("cycles"), record.cycles)
        record.stopped_reason = as_text(payload.get("reason"))
        merge_paths(record.changed_files, payload.get("changed_files"))


def collect_outcome(record: RunRecord, orchestrator: Orchestrator) -> None:
    """Copy an orchestrator's final outcome into its registry entry.

    The four attributes reported here are read with ``getattr`` so a stub engine -- or an
    orchestrator that died mid-run -- still leaves a usable record instead of raising
    inside the terminal frame.
    """
    record.cycles = as_int(getattr(orchestrator, "cycles", None), record.cycles)
    reason = as_text(getattr(orchestrator, "stopped_reason", None))
    if reason:
        record.stopped_reason = reason
    merge_paths(record.changed_files, getattr(orchestrator, "changed_files", None))
    record.tasks = [
        {
            "id": as_text(getattr(task, "id", None)),
            "status": as_text(getattr(task, "status", None)),
            "result": getattr(task, "result", None),
        }
        for task in (getattr(orchestrator, "tasks", None) or [])
    ]


async def drive(record: RunRecord, sink: FrameSink) -> None:
    """Build and run the orchestrator for ``record``, then push one terminal frame.

    Every exit -- a clean finish, a raising run, a client that left -- produces a frame the
    client can read plus the end-of-stream sentinel, and never an unhandled server error.
    """
    record.runner = asyncio.current_task()
    try:
        try:
            orchestrator = build_orchestrator(record.goal, event_hook(record, sink), record.model)
            await orchestrator.run()
        except asyncio.CancelledError:
            record.stopped_reason = record.stopped_reason or CANCELLED_REASON
            sink.put_nowait(frame("result", record.result_payload()))
            raise
        except Exception as exc:  # the client must learn why, never see a server error
            logger.exception("run %s failed", record.id)
            record.error = describe(exc)
            sink.put_nowait(error_frame(record.error))
        else:
            collect_outcome(record, orchestrator)
            sink.put_nowait(frame("result", record.result_payload()))
    finally:
        record.finished_at = time.time()
        sink.put_nowait(None)


async def pump(websocket: WebSocket, frames: asyncio.Queue[dict[str, Any] | None]) -> None:
    """Forward queued frames to the client until the run ends or the client goes away."""
    while True:
        payload = await frames.get()
        if payload is None:
            return
        try:
            await send(websocket, payload)
        except (RuntimeError, WebSocketDisconnect):
            # The socket is gone; the caller cancels the run, which is the point.
            return


async def watch_client(websocket: WebSocket, frames: FrameSink) -> None:
    """End the stream when the client disconnects or asks the run to stop."""
    try:
        while True:
            message = await websocket.receive_text()
            if is_stop(message):
                logger.info("client asked run to stop")
                break
    except (RuntimeError, WebSocketDisconnect):
        logger.debug("client went away; the in-flight run will be cancelled")
    frames.put_nowait(None)


def is_stop(message: str) -> bool:
    """True when a client message is the stop command; unparseable text is ignored."""
    try:
        decoded = json.loads(message)
    except ValueError:
        return False
    return isinstance(decoded, dict) and decoded.get("type") == STOP_COMMAND


async def send(websocket: WebSocket, payload: dict[str, Any]) -> None:
    """Send one frame as JSON text."""
    await websocket.send_text(dumps(payload))


async def close(websocket: WebSocket, code: int = 1000) -> None:
    """Close the socket, tolerating a client that already went away."""
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await websocket.close(code=code)


def dumps(payload: dict[str, Any]) -> str:
    """Serialise a frame; a payload that is not JSON-clean degrades to text."""
    return json.dumps(payload, default=str)


def describe(exc: BaseException) -> str:
    """One-line rendering of an exception, safe to show a client."""
    return f"{type(exc).__name__}: {exc}"


def as_int(value: object, default: int = 0) -> int:
    """``int(value)`` for values that may be missing or of an unexpected type."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(str(value))
    except ValueError:
        return default


def as_text(value: object) -> str | None:
    """Render an enum member or string as text; anything else becomes ``None``."""
    if value is None:
        return None
    text = value if isinstance(value, str) else getattr(value, "value", None)
    return text if isinstance(text, str) and text else None


def merge_paths(target: list[str], paths: object) -> None:
    """Append every path not listed yet, preserving first-seen order."""
    if not isinstance(paths, list):
        return
    for path in paths:
        if isinstance(path, str) and path not in target:
            target.append(path)


def cancel(*tasks: asyncio.Task[Any]) -> None:
    """Request cancellation of every task that is still running."""
    for task in tasks:
        if not task.done():
            task.cancel()


async def settle(*tasks: asyncio.Task[Any]) -> None:
    """Wait for cancelled tasks to finish, tolerating our own cancellation."""
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.gather(*tasks, return_exceptions=True)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
    <button type="button" id="stop" disabled>Stop</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const stopButton = document.getElementById('stop');
    let socket = null;

    // Server text is appended as text, never as markup, so it cannot inject HTML.
    const log = (text) => { out.textContent += text + '\\n'; };

    const describe = (payload) => {
      const data = payload.data || {};
      if (payload.type === 'event') {
        if (payload.event === 'plan') {
          return 'plan: ' + (data.tasks || []).length + ' task(s)';
        }
        if (payload.event === 'task_done') {
          return 'task ' + data.task + ': ' + data.status + (data.skipped ? ' (skipped)' : '');
        }
        return payload.event + ': ' + JSON.stringify(data);
      }
      if (payload.type === 'result') {
        return 'finished after ' + data.cycles + ' cycle(s): '
          + (data.stopped_reason || 'no reason recorded');
      }
      return 'error: ' + (data.message || 'unknown error');
    };

    form.onsubmit = (e) => {
      e.preventDefault();
      log('Starting...');
      stopButton.disabled = false;
      const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
      socket = new WebSocket(scheme + '//' + location.host + '/ws');
      socket.onopen = () => {
        socket.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
      socket.onmessage = (ev) => {
        let payload;
        try {
          payload = JSON.parse(ev.data);
        } catch (err) {
          log(String(ev.data));
          return;
        }
        log(describe(payload));
      };
      socket.onerror = () => log('websocket error');
      socket.onclose = () => {
        socket = null;
        stopButton.disabled = true;
        log('Closed.');
      };
    };

    stopButton.onclick = () => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({type: 'stop'}));
        log('Stop requested.');
      }
    };
  </script>
</body>
</html>
"""
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Stream one orchestrated run to a browser, then close.

    Every failure mode is answered with a frame rather than a traceback: a message that is
    not JSON, a message without a usable goal, an engine that raises. A client that
    disconnects part way through has its run task cancelled, not abandoned.
    """
    await websocket.accept()
    try:
        try:
            payload = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        except ValueError as exc:  # json.JSONDecodeError and friends
            await send(websocket, error_frame(f"the start message must be JSON: {exc}"))
            await close(websocket, WS_UNSUPPORTED_DATA)
            return

        goal, model = parse_start(payload)
        if goal is None:
            await send(websocket, error_frame(INVALID_GOAL_MESSAGE))
            await close(websocket, WS_POLICY_VIOLATION)
            return

        frames: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        record = register_run(goal, model)
        runner = asyncio.create_task(drive(record, frames), name=record.id)
        watcher = asyncio.create_task(watch_client(websocket, frames), name=f"{record.id}-client")
        try:
            await pump(websocket, frames)
        finally:
            cancel(runner, watcher)
            await settle(runner, watcher)
        await close(websocket)
    except WebSocketDisconnect:
        return


@app.post("/api/start", status_code=202)
async def api_start(request: StartRequest) -> dict[str, Any]:
    """Register and launch a run that no browser is watching.

    A blank goal is rejected by ``StartRequest`` itself, so this returns 422 for it; the
    caller gets the registry entry immediately and the outcome from ``GET /api/runs``.
    """
    record = register_run(request.goal, request.model)
    asyncio.create_task(drive(record, NullSink()), name=record.id)
    return record.summary()


@app.get("/api/runs")
async def api_runs() -> list[dict[str, Any]]:
    """Every retained run, oldest first."""
    return [record.summary() for record in _RUNS.values()]


@app.get("/api/runs/{run_id}")
async def api_run(run_id: str) -> dict[str, Any]:
    """One run with its task list and its recorded event trail."""
    record = _RUNS.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return record.detail()


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the monitoring UI.

    Loopback by default: there is no authentication, CORS or rate limiting here and every
    run spends money on LLM calls, so exposing the endpoint is an explicit choice -- pass
    ``host="0.0.0.0"``, ideally behind a reverse proxy that authenticates.
    """
    uvicorn.run(app, host=host, port=port)
