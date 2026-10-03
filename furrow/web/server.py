from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from furrow.core import Event, EventType, Orchestrator

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import time cheap
    from furrow.config import Settings

logger = logging.getLogger(__name__)

#: Listen address used when the caller does not pass one to :func:`run`.
DEFAULT_HOST = "0.0.0.0"
#: Listen port used when the caller does not pass one to :func:`run`.
DEFAULT_PORT = 8000

#: An orchestrator event callback may be a plain function or a coroutine function.
OnEvent = Callable[[Event], Awaitable[None] | None]

app = FastAPI(title="Furrow")


# --------------------------------------------------------------------------- models


class StartRequest(BaseModel):
    """Payload accepted by ``POST /api/run`` and by the first websocket frame."""

    goal: str = Field(min_length=1, description="What the orchestrator should achieve.")
    max_cycles: int | None = Field(
        default=None, ge=1, description="Cycle budget; omit for the configured default."
    )
    model: str | None = Field(default=None, description="Optional model override.")

    @field_validator("goal")
    @classmethod
    def _goal_must_not_be_blank(cls, value: str) -> str:
        """Reject goals that are empty or only whitespace."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("goal must not be empty")
        return cleaned


# ------------------------------------------------------------------- run registry


def _empty_state() -> dict[str, Any]:
    """Return the snapshot shape used before anything has run."""
    return {"goal": None, "cycles": 0, "done": False, "tasks": [], "history": []}


class _Run:
    """One in-flight orchestrator run plus the task that drives it."""

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        self.task: asyncio.Task[None] | None = None

    @property
    def goal(self) -> str:
        """Return the goal this run was started with."""
        return self.orchestrator.goal


#: The single run allowed at a time; a second ``/api/run`` is rejected with 409.
_active_run: _Run | None = None
#: Snapshot of the most recent run, served by ``/api/state`` when idle.
_last_state: dict[str, Any] = _empty_state()
#: Strong references to detached tasks so they cannot be garbage collected early.
_background_tasks: set[asyncio.Task[None]] = set()


def _begin_run(orchestrator: Orchestrator) -> _Run:
    """Register ``orchestrator`` as the active run."""
    global _active_run, _last_state
    _active_run = _Run(orchestrator)
    _last_state = _empty_state()
    return _active_run


def _finish_run(run: _Run, state: dict[str, Any]) -> None:
    """Release the registry slot held by ``run`` and remember its final state."""
    global _active_run, _last_state
    _last_state = state
    if _active_run is run:
        _active_run = None


def _current_state() -> dict[str, Any]:
    """Return the live snapshot of the active run, or the last one to finish."""
    run = _active_run
    if run is not None:
        return run.orchestrator.state()
    return dict(_last_state)


def _track_task(task: asyncio.Task[None]) -> None:
    """Hold a strong reference to a detached task until it completes."""
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _execute_run(orchestrator: Orchestrator, run: _Run, max_cycles: int | None) -> None:
    """Drive the loop to completion and release the registry slot when it ends."""
    try:
        await orchestrator.run(max_cycles=max_cycles)
    except asyncio.CancelledError:
        logger.info("run for goal %r cancelled", run.goal)
        raise
    except Exception:
        logger.exception("run for goal %r failed", run.goal)
    finally:
        _finish_run(run, orchestrator.state())


def _build_orchestrator(
    goal: str, on_event: OnEvent | None = None, model: str | None = None
) -> Orchestrator:
    """Create an orchestrator for ``goal``, optionally overriding the model."""
    return Orchestrator(goal=goal, settings=_resolve_settings(model), on_event=on_event)


def _resolve_settings(model: str | None) -> Settings | None:
    """Return settings with ``model`` applied, or ``None`` to use the defaults."""
    if not model:
        return None
    from furrow.config import settings as global_settings

    return global_settings.model_copy(update={"model": model})


def _frame(event_type: EventType, message: str, cycle: int = 0) -> dict[str, Any]:
    """Build a JSON frame shaped exactly like an orchestrator event."""
    return Event(type=event_type, cycle=cycle, message=message).to_dict()


@lru_cache(maxsize=1)
def _package_version() -> str:
    """Return the installed distribution version, falling back to the source value."""
    try:
        return version("furrow")
    except PackageNotFoundError:
        return "0.1.0"


# ------------------------------------------------------------------ REST surface


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Report that the server is up and which version is running."""
    return {"status": "ok", "version": _package_version()}


@app.get("/api/state")
async def api_state() -> dict[str, Any]:
    """Return the active run's snapshot, or the last completed one."""
    return _current_state()


@app.post("/api/run")
async def api_run(payload: StartRequest) -> JSONResponse:
    """Start a run in the background and return immediately."""
    if _active_run is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"a run is already in progress: {_active_run.goal}",
        )
    orchestrator = _build_orchestrator(payload.goal, model=payload.model)
    run = _begin_run(orchestrator)
    run.task = asyncio.create_task(
        _execute_run(orchestrator, run, payload.max_cycles), name="furrow-run"
    )
    _track_task(run.task)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"started": True, "goal": payload.goal, "max_cycles": payload.max_cycles},
    )


@app.post("/api/stop")
async def api_stop() -> dict[str, bool]:
    """Ask the active orchestrator to stop before its next cycle."""
    run = _active_run
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no run is in progress"
        )
    run.orchestrator.request_stop()
    return {"stopped": True}


# -------------------------------------------------------------- websocket surface


class _SocketSender:
    """Serialise websocket sends so frames can never interleave.

    The orchestrator emits events from inside its own coroutines, and those
    callbacks can fire while the endpoint itself is also writing an error
    frame, so every write goes through one lock.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._lock = asyncio.Lock()
        self._closed = False

    async def send(self, frame: dict[str, Any]) -> bool:
        """Send one frame; report whether it made it to the client."""
        if self._closed:
            return False
        async with self._lock:
            if self._closed:
                return False
            try:
                await self._websocket.send_json(frame)
            except Exception as exc:  # noqa: BLE001 - a dead socket must not break the run
                self._closed = True
                logger.info("websocket send failed: %s", exc)
                return False
        return True

    async def close(self) -> None:
        """Close the socket once, ignoring an already broken transport."""
        async with self._lock:
            self._closed = True
            with contextlib.suppress(Exception):
                await self._websocket.close()


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    """Block until the client goes away, ignoring any stray client frames."""
    while True:
        try:
            message = await websocket.receive()
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001 - transport failure counts as a disconnect
            logger.info("websocket receive failed: %s", exc)
            return
        if message.get("type") == "websocket.disconnect":
            return


async def _read_start_request(websocket: WebSocket) -> StartRequest:
    """Read and validate the first websocket frame.

    Raises ``WebSocketDisconnect`` if the client left, and ``ValueError`` with a
    human readable reason when the frame is not a valid start request.
    """
    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("first frame must be a JSON object") from exc
    try:
        return StartRequest.model_validate(data)
    except ValidationError as exc:
        raise ValueError(_validation_message(exc)) from exc


def _validation_message(exc: ValidationError) -> str:
    """Flatten pydantic errors into one short, log-friendly line."""
    problems = [
        f"{'.'.join(str(part) for part in error.get('loc', ())) or 'body'}: {error.get('msg')}"
        for error in exc.errors()
    ]
    return "; ".join(problems) or "invalid start request"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Stream orchestrator events to a browser until the run or the client ends."""
    await websocket.accept()
    sender = _SocketSender(websocket)
    run: _Run | None = None
    task: asyncio.Task[None] | None = None
    watcher: asyncio.Task[None] | None = None
    try:
        try:
            request = await _read_start_request(websocket)
        except WebSocketDisconnect:
            return
        except ValueError as exc:
            await sender.send(_frame(EventType.ERROR, f"invalid start request: {exc}"))
            return
        if _active_run is not None:
            await sender.send(
                _frame(EventType.ERROR, f"a run is already in progress: {_active_run.goal}")
            )
            return

        async def on_event(event: Event) -> None:
            await sender.send(event.to_dict())

        orchestrator = _build_orchestrator(
            request.goal, on_event=on_event, model=request.model
        )
        run = _begin_run(orchestrator)
        task = asyncio.create_task(
            _execute_run(orchestrator, run, request.max_cycles), name="furrow-ws-run"
        )
        run.task = task
        _track_task(task)
        watcher = asyncio.create_task(_wait_for_disconnect(websocket), name="furrow-ws-watch")
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if task.done() and not task.cancelled() and task.exception() is not None:
            await sender.send(_frame(EventType.ERROR, f"run failed: {task.exception()}"))
    finally:
        if watcher is not None:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watcher
        if task is not None:
            if not task.done():
                # The client vanished mid-run and the loop may never end: stop it.
                if run is not None:
                    run.orchestrator.request_stop()
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if run is not None:
            _finish_run(run, run.orchestrator.state())
        await sender.close()


# ----------------------------------------------------------------------- helpers


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Serve the console with uvicorn on ``host``/``port`` (Settings-style config)."""
    logger.info("Furrow console listening on http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


# ------------------------------------------------------------------- HTML console

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Furrow console</title>
<style>
:root { color-scheme: dark light; --fg: #e7e9ee; --bg: #14161a; --dim: #7f8c98;
        --ok: #7ee787; --err: #ff8f8f; --line: #2a2f38; --card: #1b1e25; }
* { box-sizing: border-box; }
body { margin: 0; padding: 1.5rem; background: var(--bg); color: var(--fg);
       font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
main { max-width: 62rem; margin: 0 auto; }
h1 { margin: 0 0 .3rem; font-size: 1.2rem; letter-spacing: .05em; }
p.sub { margin: 0 0 1.1rem; color: var(--dim); }
form { display: flex; gap: .5rem; flex-wrap: wrap; }
input { flex: 1 1 22rem; padding: .5rem .6rem; font: inherit; color: inherit;
        background: var(--card); border: 1px solid var(--line); border-radius: .4rem; }
button { padding: .5rem 1.1rem; font: inherit; color: inherit; cursor: pointer;
         background: var(--card); border: 1px solid var(--line); border-radius: .4rem; }
button:disabled { opacity: .4; cursor: not-allowed; }
#stop:not(:disabled) { border-color: #7a3030; }
#status, #tasks { margin: .7rem 0 0; color: var(--dim); }
#status { color: var(--fg); }
#log { list-style: none; margin: 1rem 0 0; padding: .5rem .7rem; min-height: 10rem;
       max-height: 26rem; overflow-y: auto; background: var(--card);
       border: 1px solid var(--line); border-radius: .4rem; }
#log li { padding: .1rem 0; white-space: pre-wrap; overflow-wrap: anywhere; }
#log li.error { color: var(--err); }
#log li.done { color: var(--ok); }
#log li.info { color: var(--dim); }
</style>
</head>
<body>
<main>
  <h1>Furrow</h1>
  <p class="sub">Autonomous development loop. Events stream live over the websocket.</p>
  <form id="controls">
    <input id="goal" type="text" autocomplete="off" required
           aria-label="goal" placeholder="What should Furrow build?">
    <button id="start" type="submit">Start</button>
    <button id="stop" type="button" disabled>Stop</button>
  </form>
  <p id="status">idle</p>
  <p id="tasks">tasks: none</p>
  <ol id="log"></ol>
</main>
<script>
(function () {
  'use strict';
  var form = document.getElementById('controls');
  var goalInput = document.getElementById('goal');
  var startButton = document.getElementById('start');
  var stopButton = document.getElementById('stop');
  var statusLine = document.getElementById('status');
  var taskLine = document.getElementById('tasks');
  var logList = document.getElementById('log');
  var socket = null;
  var cycle = 0;
  var tasks = Object.create(null);

  function wsUrl() {
    var scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    return scheme + '://' + location.host + '/ws';
  }

  function setRunning(running) {
    startButton.disabled = running;
    goalInput.disabled = running;
    stopButton.disabled = !running;
  }

  function render() {
    statusLine.textContent = cycle > 0 ? 'running - cycle ' + cycle : 'idle';
    var names = Object.keys(tasks);
    taskLine.textContent = 'tasks: ' + (names.length
      ? names.map(function (name) { return name + '=' + tasks[name]; }).join(', ')
      : 'none');
  }

  // Every server supplied string lands here and is written with textContent only.
  function append(kind, text) {
    var item = document.createElement('li');
    item.className = kind;
    item.textContent = '[' + new Date().toLocaleTimeString() + '] ' + text;
    logList.appendChild(item);
    logList.scrollTop = logList.scrollHeight;
  }

  function track(data) {
    var task = (data || {}).task;
    if (task && task.id) {
      tasks[task.id] = task.status || 'running';
    }
  }

  function describe(frame) {
    var data = frame.data || {};
    var task = data.task || {};
    var type = frame.type || 'event';
    if (type === 'task_start' || type === 'task_end') {
      return type + ' ' + (task.id || '?') + ': ' + (task.description || frame.message || '');
    }
    if (type === 'plan') {
      var planned = Array.isArray(data.tasks) ? data.tasks.length : 0;
      return 'plan [' + planned + ' task(s)]: ' + (frame.message || '');
    }
    if (type === 'test_end') {
      return 'tests ' + (data.passed ? 'passed' : 'failed') + ': ' + (frame.message || '');
    }
    return type + ': ' + (frame.message || '');
  }

  function onFrame(frame) {
    if (frame.cycle) { cycle = frame.cycle; }
    track(frame.data);
    if (frame.type === 'done') {
      append('done', 'done: ' + (frame.message || ''));
      render();
      release();
      return;
    }
    append(frame.type === 'error' ? 'error' : 'event', describe(frame));
    render();
  }

  function release() {
    setRunning(false);
    if (socket) {
      var closing = socket;
      socket = null;
      closing.close();
    }
  }

  function connect(goal) {
    var url = wsUrl();
    setRunning(true);
    append('info', 'connecting to ' + url);
    var ws = new WebSocket(url);
    socket = ws;
    ws.onopen = function () { ws.send(JSON.stringify({ goal: goal })); };
    ws.onmessage = function (message) {
      var frame = null;
      try { frame = JSON.parse(message.data); } catch (err) { frame = null; }
      if (frame && typeof frame === 'object') { onFrame(frame); }
      else { append('error', 'unreadable frame'); }
    };
    ws.onerror = function () { append('error', 'websocket error'); };
    ws.onclose = function () {
      if (socket !== ws) { return; }
      socket = null;
      setRunning(false);
      append('info', 'connection closed');
    };
  }

  form.addEventListener('submit', function (event) {
    event.preventDefault();
    if (socket) { return; }
    var goal = goalInput.value.trim();
    if (!goal) { append('error', 'enter a goal first'); return; }
    cycle = 0;
    tasks = Object.create(null);
    render();
    connect(goal);
  });

  stopButton.addEventListener('click', function () {
    stopButton.disabled = true;
    append('info', 'stop requested');
    fetch('/api/stop', { method: 'POST' })
      .then(function (response) {
        return response.json().then(function (body) {
          return { ok: response.ok, body: body };
        });
      })
      .then(function (result) {
        if (!result.ok) {
          append('error', 'stop refused: ' + (result.body.detail || 'unknown error'));
        }
      })
      .catch(function (err) { append('error', 'stop failed: ' + err); });
  });

  fetch('/api/state')
    .then(function (response) { return response.ok ? response.json() : null; })
    .then(function (body) {
      if (!body || !body.goal) { return; }
      cycle = body.cycles || 0;
      (body.tasks || []).forEach(function (task) {
        if (task && task.id) { tasks[task.id] = task.status || 'pending'; }
      });
      append('info', 'last run: ' + body.goal);
      render();
    })
    .catch(function () { /* nothing has run yet */ });

  render();
}());
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single page console."""
    return HTMLResponse(content=INDEX_HTML)