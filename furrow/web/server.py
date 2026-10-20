"""Web layer for Furrow.

The server exposes a tiny JSON API plus a websocket that streams every
orchestrator event to the browser:

    GET  /            single page UI (inline HTML/CSS/JS)
    GET  /api/state  persisted run state
    GET  /api/status status of the currently active run
    POST /api/start  start a background run
    WS   /ws         run a goal and stream its events
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from furrow.config import RunResult, Settings, settings
from furrow.core.events import Event, EventBus
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import StateStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")

POLL_INTERVAL = 0.05

# Rolling buffer of the most recent orchestrator events, for clients that
# connect after a run has started (or to poll history without a WebSocket).
EVENT_BUFFER: deque[Event] = deque(maxlen=500)

DEFAULT_CURRENT: dict[str, Any] = {
    "running": False,
    "orchestrator": None,
    "result": None,
    "task": None,
    "error": None,
}

# Registry of the run that is currently active (or of the last one that ran).
_CURRENT: dict[str, Any] = dict(DEFAULT_CURRENT)

# Strong references to background tasks so they are never garbage collected
# while they are still pending.
_TASKS: set[asyncio.Task[RunResult]] = set()

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow</title>
<style>
  :root { color-scheme: dark; }
  body {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    margin: 0; padding: 1.25rem; background: #10131a; color: #d7dae0;
  }
  h1 {
    font-size: 1.3rem; letter-spacing: .14em; text-transform: uppercase;
    margin: 0 0 1rem; color: #f0f2f5;
  }
  .bar { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1rem; }
  input {
    flex: 1 1 20rem; padding: .55rem .7rem; background: #171b23; color: inherit;
    border: 1px solid #2b3240; border-radius: 6px; font: inherit;
  }
  button {
    padding: .55rem 1.1rem; border-radius: 6px; border: 1px solid #2b3240;
    background: #1d2430; color: inherit; font: inherit; cursor: pointer;
  }
  button:hover { border-color: #4c8dff; }
  #log {
    background: #0b0e13; border: 1px solid #2b3240; border-radius: 6px;
    padding: .75rem; height: 20rem; overflow: auto;
    white-space: pre-wrap; word-break: break-word; margin: 0;
  }
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
  th, td {
    text-align: left; padding: .4rem .5rem; font-size: .9rem;
    border-bottom: 1px solid #2b3240; vertical-align: top;
  }
  th { color: #8a93a5; font-weight: 600; }
  .done { color: #57d38c; }
  .failed { color: #e0685f; }
  .muted { color: #8a93a5; }
</style>
</head>
<body>
<h1>Furrow</h1>
<div class="bar">
  <input id="goal" placeholder="What should Furrow build?" autofocus />
  <button id="start">Start</button>
  <button id="resume">Resume</button>
</div>
<pre id="log"></pre>
<table>
  <thead><tr><th>Task</th><th>Status</th><th>Details</th></tr></thead>
  <tbody id="tasks"></tbody>
</table>
<script>
  const logEl = document.getElementById('log');
  const goalEl = document.getElementById('goal');
  const tasksEl = document.getElementById('tasks');
  let socket = null;

  function append(line) {
    logEl.textContent += line + '\n';
    logEl.scrollTop = logEl.scrollHeight;
  }

  function text(value) {
    return value === undefined || value === null ? '' : String(value);
  }

  function renderTasks(tasks) {
    tasksEl.innerHTML = '';
    (tasks || []).forEach(function (task) {
      const row = document.createElement('tr');
      const status = text(task.status) || 'pending';
      [text(task.id), status, text(task.result || task.error || task.description)]
        .forEach(function (value) {
          const cell = document.createElement('td');
          cell.textContent = value;
          if (value === 'completed') { cell.className = 'done'; }
          if (value === 'failed') { cell.className = 'failed'; }
          row.appendChild(cell);
        });
      tasksEl.appendChild(row);
    });
  }

  async function loadState() {
    try {
      const res = await fetch('/api/state');
      const body = await res.json();
      if (body.state && body.state.tasks) { renderTasks(body.state.tasks); }
    } catch (err) {
      append('! could not load state');
    }
  }

  async function loadStatus() {
    try {
      const res = await fetch('/api/status');
      const body = await res.json();
      if (body.status && body.status.tasks) { renderTasks(body.status.tasks); }
      if (body.result) { append('-- ' + text(body.result.summary)); }
    } catch (err) {
      append('! could not load status');
    }
  }

  function connect(payload) {
    if (socket) { socket.close(); }
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(scheme + '://' + location.host + '/ws');
    socket.onopen = function () {
      append('> connected');
      socket.send(JSON.stringify(payload));
    };
    socket.onmessage = function (ev) {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch (err) {
        append(ev.data);
        return;
      }
      append(JSON.stringify(msg));
      if (msg.type === 'task_started' || msg.type === 'task_finished') {
        loadStatus();
      }
      if (msg.type === 'done') {
        append('-- ' + (msg.message || 'finished'));
        loadStatus();
      }
      if (msg.type === 'error') {
        append('!! ' + (msg.message || 'error'));
      }
    };
    socket.onclose = function () {
      append('-- closed');
      socket = null;
      loadStatus();
    };
    socket.onerror = function () {
      append('!! websocket error');
    };
  }

  document.getElementById('start').onclick = function () {
    connect({ goal: goalEl.value, resume: false });
  };
  document.getElementById('resume').onclick = function () {
    connect({ goal: goalEl.value, resume: true });
  };

  loadState();
  loadStatus();
</script>
</body>
</html>
"""


class StartRequest(BaseModel):
    """Body of the ``POST /api/start`` endpoint."""

    goal: str = ""
    max_cycles: int | None = None
    resume: bool = False


class SocketRequest(BaseModel):
    """First (and only) message a websocket client sends."""

    goal: str = ""
    max_cycles: int | None = None
    resume: bool = False


def _reset_registry() -> None:
    """Forget the active run (used by the tests)."""
    _CURRENT.clear()
    _CURRENT.update(DEFAULT_CURRENT)


def _state_store() -> StateStore:
    """Store for the persisted run state of the current settings."""
    return StateStore(settings.state_path())


def _run_settings(max_cycles: int | None) -> Settings:
    """Settings for a run, with an optional ``max_cycles`` override."""
    if max_cycles is None:
        return settings
    return settings.model_copy(update={"max_cycles": max_cycles})


def _on_run_done(task: asyncio.Task[RunResult]) -> None:
    """Finalise the registry when a run ends."""
    _TASKS.discard(task)
    if _CURRENT.get("task") is not task:
        return
    _CURRENT["running"] = False
    if task.cancelled():
        _CURRENT["error"] = "run cancelled"
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Furrow run failed: %s", exc)
        _CURRENT["error"] = str(exc)
        return
    _CURRENT["result"] = task.result()


async def _drain(queue: asyncio.Queue[Event], buffer: deque[Event] | None = None) -> None:
    """Consume events so a background run does not fill its queue forever.

    Optionally appends each event to a rolling buffer so recent history is
    available via ``GET /api/events`` after the fact.
    """
    while True:
        event = await queue.get()
        if buffer is not None:
            buffer.append(event)


async def _run_and_drain(orchestrator: Orchestrator, bus: EventBus) -> RunResult:
    """Run an orchestrator in the background and buffer its emitted events."""
    queue = bus.subscribe()
    drain = asyncio.create_task(_drain(queue, EVENT_BUFFER))
    try:
        return await orchestrator.run()
    finally:
        drain.cancel()
        bus.unsubscribe(queue)


def _build_orchestrator(
    goal: str, resume: bool, max_cycles: int | None, bus: EventBus
) -> Orchestrator:
    """Create an orchestrator wired to the event bus and the state store."""
    return Orchestrator(
        goal=goal,
        settings=_run_settings(max_cycles),
        state_store=StateStore(settings.state_path()),
        event_bus=bus,
        resume=resume,
    )


async def _send_error(websocket: WebSocket, message: str) -> None:
    """Send an error frame to the client and close the connection."""
    await websocket.send_json({"type": "error", "message": message, "data": {}})
    await websocket.close(code=1008)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """The single page UI."""
    return HTMLResponse(content=INDEX_HTML)


@app.get("/api/state")
async def api_state() -> dict[str, Any]:
    """The persisted run state, if there is one."""
    store = _state_store()
    state = store.load()
    return {
        "state": state.model_dump(mode="json") if state is not None else None,
        "exists": store.exists(),
    }


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    """Status of the run that is currently active."""
    orchestrator = _CURRENT.get("orchestrator")
    result = _CURRENT.get("result")
    return {
        "running": bool(_CURRENT.get("running")),
        "status": orchestrator.status() if orchestrator is not None else None,
        "result": result.model_dump(mode="json") if result is not None else None,
    }


@app.get("/api/events")
async def api_events() -> dict[str, Any]:
    """Recent orchestrator events from the rolling buffer (most recent last)."""
    return {
        "events": [event.to_dict() for event in EVENT_BUFFER],
    }


@app.post("/api/start")
async def api_start(request: StartRequest) -> JSONResponse:
    """Start a background run for the given goal."""
    if _CURRENT.get("running"):
        return JSONResponse(
            status_code=409,
            content={"started": False, "error": "a run is already in progress"},
        )
    goal = request.goal.strip()
    if not goal and not request.resume:
        return JSONResponse(
            status_code=400,
            content={"started": False, "error": "goal is required"},
        )
    if not goal:
        state = _state_store().load()
        goal = state.original_goal if state is not None else ""
        if not goal:
            return JSONResponse(
                status_code=400,
                content={"started": False, "error": "no goal to resume"},
            )
    bus = EventBus()
    orchestrator = _build_orchestrator(goal, request.resume, request.max_cycles, bus)
    run_task = asyncio.create_task(_run_and_drain(orchestrator, bus))
    _TASKS.add(run_task)
    run_task.add_done_callback(_on_run_done)
    _CURRENT.update(
        {
            "running": True,
            "orchestrator": orchestrator,
            "result": None,
            "task": run_task,
            "error": None,
        }
    )
    return JSONResponse(status_code=200, content={"started": True, "goal": goal})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Run a goal and stream every event it emits to the client."""
    await websocket.accept()
    bus = EventBus()
    queue = bus.subscribe()
    run_task: asyncio.Task[RunResult] | None = None
    try:
        try:
            payload = await websocket.receive_json()
        except (WebSocketDisconnect, RuntimeError, ValueError):
            return
        try:
            request = SocketRequest.model_validate(payload)
        except ValueError:
            await _send_error(websocket, "expected a JSON object with a goal")
            return
        goal = request.goal.strip()
        resume = request.resume
        if not goal and not resume:
            await _send_error(websocket, "goal is required")
            return
        orchestrator = _build_orchestrator(goal, resume, request.max_cycles, bus)
        run_task = asyncio.create_task(orchestrator.run())
        _TASKS.add(run_task)
        run_task.add_done_callback(_on_run_done)
        _CURRENT.update(
            {
                "running": True,
                "orchestrator": orchestrator,
                "result": None,
                "task": run_task,
                "error": None,
            }
        )
        while True:
            if run_task.done() and queue.empty():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=POLL_INTERVAL)
            except asyncio.TimeoutError:
                continue
            await websocket.send_json(event.to_dict())
        if run_task.cancelled():
            return
        exc = run_task.exception()
        if exc is not None:
            logger.error("Furrow run failed: %s", exc)
            await _send_error(websocket, f"run failed: {exc}")
            return
        result = run_task.result()
        await websocket.send_json(
            {
                "type": "done",
                "message": result.summary,
                "data": result.model_dump(mode="json"),
            }
        )
    except (WebSocketDisconnect, RuntimeError):
        if run_task is not None and not run_task.done():
            run_task.cancel()
    finally:
        bus.unsubscribe(queue)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the Furrow web server."""
    uvicorn.run(app, host=host, port=port)
