"""FastAPI application exposing a Furrow run over HTTP and a WebSocket.

Two ways to start a run are offered. ``POST /api/start`` schedules a run and
returns ``202`` immediately, which is what scripts and health checks want.
The ``/ws`` endpoint is what the bundled page uses: it accepts any number of
``{"goal": ..., "model": ...}`` frames on one connection, runs them one at a
time in the order they arrived, and streams the orchestrator's events back as
they happen, then a ``run_end`` frame carrying the
:class:`~furrow.core.events.RunResult`. Runs are serialised per connection, but
a frame sent *while* a run is streaming is queued and starts its own run once
the current one ends, so nothing a client sends is ever dropped.

Security
--------
Furrow is an agent that writes and executes code in the workspace it is pointed
at. This server ships **no authentication**, so it binds ``127.0.0.1`` by
default. Binding any other interface (``0.0.0.0``, a LAN address, a container
bridge) publishes an unauthenticated remote code execution surface to everyone
who can reach the port. :func:`run` prints a loud warning when asked to do so,
but the exposure is the operator's decision to make.

Events are bridged to the socket through an :class:`asyncio.Queue`: the
``on_event`` callback is synchronous and may run inside async code, so it only
calls ``put_nowait`` and a separate task does the awaiting. Client frames travel
the same way in the other direction: one long-lived reader task owns every
``websocket.receive()`` for the life of the connection and queues each frame the
instant it lands, which is the only way to receive without a window in which an
arriving frame can be destroyed by a cancelled read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import suppress
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from furrow.config import Settings, configure_logging, get_settings
from furrow.core.events import FurrowEvent, RunResult
from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "127.0.0.0/8"})


class StartRequest(BaseModel):
    """Body of ``POST /api/start`` and of every client frame sent to ``/ws``."""

    goal: str = Field(
        min_length=1, description="What the agent should try to accomplish."
    )
    model: str | None = Field(
        default=None, description="Model override for every role; null keeps the configured ones."
    )


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: str = "ok"


class StartAccepted(BaseModel):
    """Body returned by ``POST /api/start`` once a run has been scheduled."""

    run_id: str
    goal: str
    status: str = "accepted"
    stream_path: str = "/ws"


class EventFrame(BaseModel):
    """One streamed event, forwarded to the browser verbatim plus a run id."""

    type: str
    message: str = ""
    cycle: int = 0
    task_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    run_id: str = ""

    @classmethod
    def from_event(cls, event: FurrowEvent, run_id: str) -> EventFrame:
        """Wrap an orchestrator event for transmission.

        Args:
            event: The event emitted by the orchestrator.
            run_id: Identifier of the run that produced it.

        Returns:
            A frame carrying every field of `event` plus the run id.
        """
        payload = event.model_dump()
        payload["run_id"] = run_id
        return cls.model_validate(payload)


class RunEndFrame(BaseModel):
    """Terminal frame describing how a run ended."""

    type: str = "run_end"
    completed: bool = False
    run_id: str = ""
    cycles: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    cycles_failed: int = 0
    last_error: str | None = None
    stopped_reason: str = ""

    @classmethod
    def from_result(cls, result: RunResult, run_id: str) -> RunEndFrame:
        """Wrap a :class:`RunResult` for transmission.

        Args:
            result: The value returned by ``Orchestrator.run``.
            run_id: Identifier of the run that produced it.

        Returns:
            A frame describing the terminal state of the run.
        """
        payload = result.model_dump()
        payload["run_id"] = run_id
        return cls.model_validate(payload)


class ErrorFrame(BaseModel):
    """Terminal error frame sent when a run or a client message goes wrong."""

    type: str = "error"
    message: str
    run_id: str = ""


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
         margin: 0 auto; max-width: 60rem; padding: 1.5rem; line-height: 1.45; }
  h1 { margin: 0 0 0.25rem; }
  .muted { opacity: 0.7; }
  form { display: flex; gap: 0.5rem; margin: 1rem 0; flex-wrap: wrap; }
  input { flex: 1 1 20rem; padding: 0.5rem; font: inherit; }
  button { padding: 0.5rem 1rem; font: inherit; cursor: pointer; }
  #status { padding: 0.5rem; border-left: 4px solid currentColor; margin: 0 0 1rem; }
  #status.busy { color: #b58900; }
  #status.ok { color: #2f855a; }
  #status.error { color: #c53030; }
  #status.idle { color: #718096; }
  .cols { display: flex; gap: 1.5rem; flex-wrap: wrap; }
  .cols > * { flex: 1 1 18rem; min-width: 0; }
  ul { list-style: none; padding-left: 0; margin: 0; }
  #log { max-height: 26rem; overflow-y: auto; border: 1px solid currentColor;
         padding: 0.5rem; white-space: pre-wrap; opacity: 0.95; }
  .task_start { color: #3182ce; }
  .task_complete { color: #2f855a; }
  .task_failed { color: #c53030; }
  .task_skipped { color: #b7791f; }
  .error { color: #c53030; font-weight: bold; }
  .cycle_start, .cycle_end, .run_start, .run_end { font-weight: bold; }
  #summary { margin-top: 1rem; }
  #summary td { padding: 0 1rem 0 0; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <p class="muted">Autonomous coding agent. Events stream live over the WebSocket.</p>
  <form id="form">
    <input id="goal" placeholder="Describe the goal" required />
    <input id="model" placeholder="Model (optional)" size="20" />
    <button id="start" type="submit">Start</button>
    <button id="stop" type="button" disabled>Stop</button>
  </form>
  <div id="status" class="idle">Idle. Enter a goal and press Start.</div>
  <div class="cols">
    <div>
      <h2>Running tasks</h2>
      <ul id="tasks"><li class="muted">none</li></ul>
      <div id="summary"></div>
    </div>
    <div>
      <h2>Event log</h2>
      <div id="log"></div>
    </div>
  </div>
<script>
const goalInput = document.getElementById('goal');
const modelInput = document.getElementById('model');
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const statusEl = document.getElementById('status');
const tasksEl = document.getElementById('tasks');
const logEl = document.getElementById('log');
const summaryEl = document.getElementById('summary');
const running = new Map();
let socket = null;

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = cls;
}

function append(frame) {
  const line = document.createElement('div');
  const type = frame.type || 'event';
  const cycle = frame.cycle ? 'cycle ' + frame.cycle + ' ' : '';
  const task = frame.task_id ? ' [' + frame.task_id + ']' : '';
  line.className = type;
  line.textContent = cycle + type + task + ' ' + (frame.message || '');
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function renderTasks() {
  tasksEl.textContent = '';
  if (running.size === 0) {
    const none = document.createElement('li');
    none.className = 'muted';
    none.textContent = 'none';
    tasksEl.appendChild(none);
    return;
  }
  running.forEach((text, id) => {
    const item = document.createElement('li');
    item.textContent = id + ': ' + text;
    tasksEl.appendChild(item);
  });
}

function renderSummary(frame) {
  summaryEl.textContent = '';
  const rows = [
    ['cycles', frame.cycles],
    ['tasks completed', frame.tasks_completed],
    ['tasks failed', frame.tasks_failed],
    ['reason', frame.stopped_reason || frame.last_error || '-']
  ];
  const table = document.createElement('table');
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    const key = document.createElement('td');
    key.textContent = row[0];
    const value = document.createElement('td');
    value.textContent = String(row[1]);
    tr.appendChild(key);
    tr.appendChild(value);
    table.appendChild(tr);
  });
  summaryEl.appendChild(table);
}

function onFrame(frame) {
  const type = frame.type || 'event';
  if (type === 'task_start') {
    running.set(frame.task_id, frame.message || 'working');
  } else if (type === 'task_complete' || type === 'task_failed' ||
             type === 'task_skipped') {
    running.delete(frame.task_id);
  } else if (type === 'run_end') {
    running.clear();
    renderSummary(frame);
    setStatus(frame.completed ? 'Completed.' : 'Finished without completing the goal.',
              frame.completed ? 'ok' : 'error');
    stopBtn.disabled = true;
  } else if (type === 'error') {
    running.clear();
    setStatus('Error: ' + (frame.message || 'unknown error'), 'error');
  } else if (type === 'run_start') {
    setStatus('Running...', 'busy');
  }
  renderTasks();
  append(frame);
}

function openSocket(goal, model) {
  const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
  socket = new WebSocket(scheme + location.host + '/ws');
  socket.onopen = () => {
    socket.send(JSON.stringify({ goal: goal, model: model }));
  };
  socket.onmessage = (event) => {
    let frame = null;
    try {
      frame = JSON.parse(event.data);
    } catch (err) {
      frame = { type: 'error', message: 'unparseable frame from server' };
    }
    onFrame(frame);
  };
  socket.onerror = () => {
    setStatus('WebSocket error: is the server still running?', 'error');
  };
  socket.onclose = () => {
    socket = null;
    stopBtn.disabled = true;
    running.clear();
    renderTasks();
    if (statusEl.className === 'busy') {
      setStatus('Connection closed; the run was cancelled.', 'error');
    }
  };
}

document.getElementById('form').onsubmit = (event) => {
  event.preventDefault();
  const goal = goalInput.value.trim();
  if (!goal) {
    setStatus('Enter a goal first.', 'error');
    return;
  }
  if (socket) {
    setStatus('A run is already in progress.', 'error');
    return;
  }
  logEl.textContent = '';
  summaryEl.textContent = '';
  running.clear();
  renderTasks();
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus('Connecting...', 'busy');
  openSocket(goal, modelInput.value.trim() || null);
};

stopBtn.onclick = () => {
  if (socket) {
    setStatus('Stopping: closing the socket cancels the run.', 'busy');
    socket.close();
  }
};

startBtn.disabled = false;
setStatus('Idle. Enter a goal and press Start.', 'idle');
</script>
</body>
</html>
"""


def _next_run_id() -> str:
    """Return a short, unique identifier for a run."""
    return f"run-{uuid4().hex[:12]}"


def _settings_for(payload: StartRequest) -> Settings:
    """Build the settings for a single run.

    A deep copy is used so two concurrent runs cannot trample each other's
    model override through the shared singleton.

    Args:
        payload: The client-supplied start request.

    Returns:
        An isolated settings instance honouring ``payload.model``.
    """
    settings = get_settings().model_copy(deep=True)
    if payload.model:
        settings.override_model = payload.model
    return settings


@app.get("/health")
async def health() -> HealthResponse:
    """Report that the process is up; no orchestration is involved."""
    return HealthResponse()


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Return the single-page UI that drives a run over ``/ws``."""
    return HTMLResponse(content=INDEX_HTML)


@app.post("/api/start", status_code=202, response_model=StartAccepted)
async def api_start(payload: StartRequest, background: BackgroundTasks) -> StartAccepted:
    """Schedule a run in the background and return immediately.

    Args:
        payload: The goal and optional model override.
        background: FastAPI's background task queue.

    Returns:
        A ``202`` body carrying the run id; the run itself is not awaited.
    """
    goal = payload.goal.strip()
    if not goal:
        raise HTTPException(status_code=422, detail="goal must not be empty")
    run_id = _next_run_id()
    orchestrator = Orchestrator(goal=goal, settings=_settings_for(payload), on_event=None)
    background.add_task(_run_unobserved, run_id, orchestrator)
    return StartAccepted(run_id=run_id, goal=goal)


async def _run_unobserved(run_id: str, orchestrator: Orchestrator) -> None:
    """Run an orchestrator that has no event consumer.

    ``POST /api/start`` starts a run nobody is watching, so an exception would
    otherwise become an unretrieved task exception.

    Args:
        run_id: Identifier used in log messages.
        orchestrator: The configured orchestrator to run.
    """
    try:
        await orchestrator.run()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("background run %s failed", run_id)


class _ClientFault:
    """A client frame the reader could not turn into a start request.

    The reader cannot raise: an exception there would tear the connection down
    with nothing telling the client why. Faults therefore travel through the
    inbound queue as values and the endpoint converts them into ``error`` frames.

    Attributes:
        message: The human-readable reason, reported verbatim in the frame.
    """

    def __init__(self, message: str) -> None:
        """Record why the frame was rejected.

        Args:
            message: The human-readable reason.
        """
        self.message = message


class _Disconnect:
    """A peer that closed the socket, announced as a queue value.

    Attributes:
        code: The WebSocket close code reported by the transport.
    """

    def __init__(self, code: int = 1000) -> None:
        """Record the close code.

        Args:
            code: The WebSocket close code reported by the transport.
        """
        self.code = code


#: The canonical disconnect sentinel, used when the transport reports no code.
_DISCONNECT = _Disconnect()


async def _read_frames(
    websocket: WebSocket,
    inbound: asyncio.Queue[Any],
    disconnected: asyncio.Event,
) -> None:
    """Move every client frame onto `inbound` the moment it arrives.

    This task owns the socket's receive side for the whole life of the
    connection. AnyIO hands an arriving frame straight to whichever receiver is
    waiting, through that receiver's private buffer, so cancelling a receive can
    take the frame with it: a read is never safely stopped by cancelling the
    task that performs it. One long-lived reader removes the hazard entirely —
    there is no receive in flight to cancel while a run streams, frames become
    durable in an unbounded queue the instant they land, and the frames that
    arrive mid-run are simply consumed, in order, after that run ends.

    Args:
        websocket: The accepted client connection.
        inbound: Unbounded queue receiving payloads, faults and disconnects.
        disconnected: Set as soon as the peer goes away, so an in-flight run can
            be cancelled without waiting for another frame to arrive.

    Returns:
        Never returns normally. The loop ends on ``websocket.disconnect``, on
        cancellation from the endpoint's cleanup, or after a read failure, and
        every one of those is reported through `inbound` and `disconnected`
        rather than by raising at the consumer.
    """
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                code = message.get("code")
                inbound.put_nowait(_Disconnect(code) if isinstance(code, int) else _DISCONNECT)
                disconnected.set()
                return
            raw = message.get("text")
            if raw is None and message.get("bytes") is not None:
                raw = message["bytes"].decode("utf-8", "replace")
            if raw is None:
                continue
            try:
                inbound.put_nowait(json.loads(raw))
            except ValueError as exc:
                inbound.put_nowait(_ClientFault(f"client sent malformed JSON: {exc}"))
    except WebSocketDisconnect:
        # Starlette raises rather than returning when the transport has already
        # reported the close; report it exactly like the returned form.
        inbound.put_nowait(_DISCONNECT)
        disconnected.set()
    except asyncio.CancelledError:
        # The endpoint is tearing the connection down: there is nobody left to
        # tell, and letting the error escape would only mean an unretrieved task
        # exception during shutdown.
        logger.debug("websocket reader cancelled")
    except Exception:
        logger.exception("websocket reader failed")
        inbound.put_nowait(_DISCONNECT)
        disconnected.set()


async def _stream_run(
    websocket: WebSocket,
    disconnected: asyncio.Event,
    run_id: str,
    task: asyncio.Task[RunResult],
    queue: asyncio.Queue[FurrowEvent],
) -> RunResult:
    """Forward queued events to the socket until `task` finishes.

    Three things are raced: the event queue, the run task, and the connection's
    disconnect flag. Racing the flag is what makes a closed tab stop the API spend
    even when the run is silent for minutes. Frames the client sends during the
    run are deliberately *not* read here — the connection's reader task queues
    them — so this loop never has to be torn down and restarted, and nothing it
    ignores can be lost.

    Args:
        websocket: The live client connection, used for sending.
        disconnected: Flag set by the reader once the peer has gone away.
        run_id: Identifier stamped onto every frame.
        task: The running orchestrator.
        queue: Queue fed by the orchestrator's ``on_event`` callback.

    Returns:
        The result of the completed run.

    Raises:
        WebSocketDisconnect: If the client closed the connection mid-run.
    """
    pending: asyncio.Future[FurrowEvent] | None = None
    closing: asyncio.Future[Any] | None = None
    try:
        while not task.done():
            pending = asyncio.ensure_future(queue.get())
            closing = asyncio.ensure_future(disconnected.wait())
            waiters: set[asyncio.Future[Any]] = {pending, closing, task}
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if closing in done and not closing.cancelled():
                closing.result()
                raise WebSocketDisconnect(_DISCONNECT.code)
            if pending in done and not pending.cancelled():
                event = pending.result()
                frame = EventFrame.from_event(event, run_id).model_dump(mode="json")
                await websocket.send_json(frame)
            elif pending is not None and not pending.done():
                pending.cancel()
            pending = None
        while not queue.empty():
            event = queue.get_nowait()
            frame = EventFrame.from_event(event, run_id).model_dump(mode="json")
            await websocket.send_json(frame)
        return task.result()
    finally:
        if pending is not None:
            pending.cancel()
        if closing is not None:
            closing.cancel()


async def _cancel_task(task: asyncio.Task[Any] | None) -> None:
    """Cancel a task and wait for it to unwind, whatever it raises.

    Args:
        task: The task to cancel, or ``None`` when there is nothing to stop.
    """
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


async def _cancel_run(task: asyncio.Task[RunResult] | None) -> None:
    """Cancel a running orchestrator and wait for it to unwind.

    Args:
        task: The task to cancel, or ``None`` when nothing is running.
    """
    await _cancel_task(task)


async def _send_error(websocket: WebSocket, message: str, run_id: str = "") -> None:
    """Send a terminal error frame and close the connection.

    Args:
        websocket: The live client connection, possibly already gone.
        message: Human-readable failure description.
        run_id: Identifier of the run that failed, when known.
    """
    frame = ErrorFrame(message=message, run_id=run_id)
    with suppress(Exception):
        await websocket.send_json(frame.model_dump(mode="json"))
    with suppress(Exception):
        await websocket.close(code=1011)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Stream runs to a browser over one long-lived connection.

    A dedicated reader task owns every receive from the moment the connection is
    accepted, so a ``{"goal": ..., "model": ...}`` frame is queued the instant it
    lands, wherever it lands: idle, mid-run, or in the instant between one run's
    ``run_end`` and the next. Runs are serialised per connection, so a frame
    received during a run simply starts its run once the current one ends. For
    each frame an orchestrator runs as a task while its events are drained from a
    queue and sent back as JSON, followed by a ``run_end`` frame. Disconnecting
    cancels the run so a closed tab stops the API spend, and no failure is
    allowed to escape as a 500.

    Args:
        websocket: The accepted client connection.
    """
    await websocket.accept()
    inbound: asyncio.Queue[Any] = asyncio.Queue()
    disconnected = asyncio.Event()
    reader = asyncio.create_task(
        _read_frames(websocket, inbound, disconnected),
        name="ws-reader",
    )
    active: asyncio.Task[RunResult] | None = None
    run_id = ""
    try:
        while True:
            item = await inbound.get()
            if isinstance(item, _Disconnect):
                raise WebSocketDisconnect(item.code)
            if isinstance(item, _ClientFault):
                raise ValueError(item.message)
            payload = StartRequest.model_validate(item)
            goal = payload.goal.strip()
            if not goal:
                raise ValueError("goal must not be empty")
            run_id = _next_run_id()
            queue: asyncio.Queue[FurrowEvent] = asyncio.Queue()
            orchestrator = Orchestrator(
                goal=goal,
                settings=_settings_for(payload),
                on_event=queue.put_nowait,
            )
            active = asyncio.create_task(orchestrator.run(), name=run_id)
            result = await _stream_run(websocket, disconnected, run_id, active, queue)
            active = None
            frame = RunEndFrame.from_result(result, run_id).model_dump(mode="json")
            await websocket.send_json(frame)
    except WebSocketDisconnect:
        logger.debug("websocket client disconnected")
    except ValueError as exc:
        # A malformed or invalid client frame is the client's mistake, not a
        # server fault: report it without a traceback and close politely.
        logger.warning("rejected websocket message: %s", exc)
        await _send_error(websocket, str(exc), run_id)
    except Exception as exc:
        logger.exception("websocket run %s failed", run_id)
        await _send_error(websocket, f"{type(exc).__name__}: {exc}", run_id)
    finally:
        await _cancel_run(active)
        await _cancel_task(reader)


def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    log_level: str | None = None,
) -> None:
    """Start the uvicorn server that serves the UI and the WebSocket.

    Args:
        host: Interface to bind. Loopback by default; anything else publishes
            an unauthenticated agent that can run code in the workspace.
        port: TCP port to listen on.
        reload: Restart on source changes. Development only.
        log_level: Verbosity for both this package's logger and uvicorn's, or
            ``None`` to use the configured ``FURROW_LOG_LEVEL``.

    Raises:
        ValueError: If `log_level` is not one of
            :data:`~furrow.config.LOG_LEVELS`.
    """
    _warn_if_public(host)
    level = configure_logging(log_level)
    uvicorn.run(app, host=host, port=port, reload=reload, log_level=level)


def _warn_if_public(host: str) -> None:
    """Print a loud warning when the server is bound beyond the local machine.

    Args:
        host: The interface that will be bound.
    """
    if host in LOOPBACK_HOSTS:
        return
    print(
        "\n"
        "============================================================\n"
        f"WARNING: binding {host} exposes Furrow on the network.\n"
        "This server has NO authentication and runs an agent that\n"
        "executes code in the workspace. Anyone who can reach the port\n"
        "can run commands on this machine. Use a tunnel or bind\n"
        "127.0.0.1 unless you have arranged your own access control.\n"
        "============================================================\n",
        file=sys.stderr,
    )


if __name__ == "__main__":
    run()
