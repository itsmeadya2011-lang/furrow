"""HTTP + websocket front end for a single Furrow run.

The websocket protocol is deliberately dumb: the client sends one of

* ``{"goal": "...", "model": "..."}``  -- start a run
* ``{"action": "stop"}``                -- ask the running orchestrator to stop

and the server pushes one JSON text frame per orchestrator event, shaped like
``{"kind": "run_start", "cycle": 0, ...}``.  Frames the server itself produces
use the same envelope (``error``, ``dropped``, ``stop_acknowledged``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketDisconnected

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
# Test seam.  Tests monkeypatch this with a fake orchestrator factory; the
# handler must therefore call it as ``_orchestrator_factory(goal=..., on_event=...)``.
_orchestrator_factory: Callable[..., Orchestrator] = Orchestrator

# Single-active-run guard.  A module level ``asyncio.Lock`` cannot serialize
# across event loops (every ``TestClient`` drives its own loop), so the guard is
# a plain module level variable that is checked and assigned without awaiting
# in between.  ``None`` means "no run in flight".
_active_run: dict[str, Any] | None = None

# State of the most recently finished run, reported by ``/api/status``.
_last_state: dict[str, Any] | None = None

# Bound on the number of events buffered for a single client.  A client that
# cannot keep up never blocks the orchestrator loop: events are dropped and
# reported with a ``{"kind": "dropped", "count": N}`` frame.
MAX_PENDING_EVENTS = 512

# How long the websocket handler waits for a stopped run to unwind before it
# cancels it.  Guarantees no orphaned run survives a disconnect.
STOP_TIMEOUT = 30.0


class StartRequest(BaseModel):
    """Payload accepted on ``/ws`` to start a run."""

    goal: str
    model: str | None = None


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------
def _task_counts(state: dict[str, Any] | None) -> dict[str, int]:
    counts = {"total": 0, "completed": 0, "failed": 0, "unfinished": 0}
    if not state:
        return counts
    tasks = state.get("tasks")
    if isinstance(tasks, int):
        counts["total"] = tasks
    elif isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            counts["total"] += 1
            status = str(task.get("status", ""))
            if status == "completed":
                counts["completed"] += 1
            elif status == "failed":
                counts["failed"] += 1
            else:
                counts["unfinished"] += 1
        return counts
    for key in counts:
        value = state.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            counts[key] = value
    return counts


def _orchestrator_state(orchestrator: Orchestrator | None) -> dict[str, Any]:
    if orchestrator is None:
        return {}
    state = getattr(orchestrator, "state", None)
    return state if isinstance(state, dict) else {}


def _result_state(task: asyncio.Task[Any]) -> dict[str, Any]:
    """Best effort state dict for a finished run task."""
    if task.cancelled():
        return {"stopped_reason": "cancelled", "cycles": 0, "tasks": []}
    error = task.exception()
    if error is not None:
        return {"stopped_reason": f"error: {error}", "cycles": 0, "tasks": []}
    result = task.result()
    return result if isinstance(result, dict) else {}


def _status() -> dict[str, Any]:
    active = _active_run
    state: dict[str, Any] | None = None
    running = False
    goal: str | None = None
    if active is not None:
        running = bool(active.get("running", True))
        raw_goal = active.get("goal")
        goal = str(raw_goal) if raw_goal else None
        candidate = active.get("state")
        if isinstance(candidate, dict):
            state = candidate
    if not running:
        state = _last_state
        raw_goal = (state or {}).get("original_goal") or (state or {}).get("goal")
        goal = goal or (str(raw_goal) if raw_goal else None)
    raw_cycles = (state or {}).get("cycles")
    cycles = raw_cycles if isinstance(raw_cycles, int) and not isinstance(raw_cycles, bool) else 0
    raw_reason = (state or {}).get("stopped_reason")
    reason = str(raw_reason) if running is False and raw_reason else None
    return {
        "running": running,
        "goal": goal,
        "cycles": cycles,
        "tasks": _task_counts(state),
        "stopped_reason": reason,
    }


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------
@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    """Current run status: ``running``, ``goal``, ``cycles``, task counts, reason."""
    return _status()


_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Furrow</title>
<style>
body { font-family: ui-monospace, monospace; margin: 2rem; }
#out { white-space: pre-wrap; border: 1px solid #ccc; padding: 1rem; min-height: 12rem; }
button { margin-right: .5rem; }
button:disabled { opacity: .5; }
</style>
</head>
<body>
<h1>Furrow</h1>
<form id="form">
  <input id="goal" name="goal" placeholder="Enter goal" required />
  <button id="start" type="submit">Start</button>
  <button id="stop" type="button" disabled>Stop</button>
</form>
<pre id="out"></pre>
<script>
const form = document.getElementById('form');
const goalInput = document.getElementById('goal');
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const out = document.getElementById('out');
const NL = '\\n';
let ws = null;

function log(text) { out.textContent += text + NL; }

function wsUrl() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  return scheme + '://' + location.host + '/ws';
}

function setRunning(running) {
  startBtn.disabled = running;
  stopBtn.disabled = !running;
}

function describe(ev) {
  const k = ev.kind || 'event';
  const id = ev.task_id || ev.id || '';
  if (k === 'plan') {
    const tasks = (ev.tasks || []).map((t) => (t.id || '?') + ': ' + (t.description || ''));
    return '[plan] ' + tasks.join(' | ');
  }
  if (k === 'task_start') return '[task] start ' + id + ' ' + (ev.description || '');
  if (k === 'task_end') return '[task] ' + (ev.status || 'done') + ' ' + id;
  if (k === 'test_result') {
    return '[tests] ' + (ev.passed ? 'pass' : 'FAIL') + ' ' + (ev.summary || '');
  }
  if (k === 'cycle_start' || k === 'cycle_end') return '[cycle ' + (ev.cycle || 0) + '] ' + k;
  if (k === 'stop_requested' || k === 'stop_acknowledged') return '[stop] requested';
  if (k === 'dropped') return '[dropped] ' + ev.count + ' event(s) (slow client)';
  if (k === 'error') return '[error] ' + (ev.message || 'unknown error');
  return '[' + k + '] ' + JSON.stringify(ev);
}

function handle(ev) {
  log(describe(ev));
  if (ev.kind === 'run_end') {
    setRunning(false);
    refresh();
  }
}

async function refresh() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    out.dataset.status = JSON.stringify(data);
  } catch (err) {
    log('[error] status: ' + err);
  }
}

function connect() {
  ws = new WebSocket(wsUrl());
  ws.onopen = () => log('Connected to ' + wsUrl());
  ws.onmessage = (ev) => {
    let payload = ev.data;
    try {
      payload = JSON.parse(ev.data);
    } catch (err) {
      log('[raw] ' + ev.data);
      return;
    }
    handle(payload);
  };
  ws.onclose = () => {
    log('Disconnected.');
    setRunning(false);
  };
  ws.onerror = () => log('WebSocket error.');
}

form.onsubmit = (ev) => {
  ev.preventDefault();
  const goal = goalInput.value.trim();
  if (!goal) {
    log('[error] goal must not be empty');
    return;
  }
  connect();
  ws.send(JSON.stringify({ goal: goal }));
  setRunning(true);
  log('Starting: ' + goal);
  refresh();
};

stopBtn.onclick = () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'stop' }));
    log('Stop requested.');
  } else {
    log('[error] not connected');
  }
};

refresh();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=_HTML)


# ---------------------------------------------------------------------------
# Event relay
# ---------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    """Coerce arbitrary payloads into something ``json.dumps`` accepts."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


def _never_raises(inner: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
    """Wrap a callback so that a failure inside it cannot break the run loop."""

    def callback(event: dict[str, Any]) -> None:
        try:
            inner(event)
        except Exception:
            return

    return callback


class _EventRelay:
    """Bounded, non-blocking fan-out from orchestrator events to one client.

    The orchestrator calls the relay synchronously from inside its loop, so the
    relay must never await and never block: it only does ``put_nowait`` on a
    bounded queue and drops events (with a marker) when the client is slow.
    """

    def __init__(self, websocket: WebSocket, maxsize: int = MAX_PENDING_EVENTS) -> None:
        self._websocket = websocket
        self._queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=maxsize)
        self._dropped = 0
        self._closed = False

    def __call__(self, event: dict[str, Any]) -> None:
        """Queue one event as a JSON text frame. Never raises."""
        if self._closed:
            return
        try:
            frame = json.dumps(_jsonable(event))
        except (TypeError, ValueError):
            frame = json.dumps({"kind": "error", "message": "event is not serializable"})
        marker: str | None = None
        if self._dropped:
            marker = json.dumps({"kind": "dropped", "count": self._dropped})
            self._dropped = 0
        for text in (marker, frame) if marker else (frame,):
            try:
                self._queue.put_nowait(text)
            except asyncio.QueueFull:
                self._dropped += 1

    async def pump(self) -> None:
        """Send queued frames until the socket dies or the relay is closed."""
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            try:
                await self._websocket.send_text(frame)
            except (WebSocketDisconnect, WebSocketDisconnected, RuntimeError, OSError):
                self._closed = True
                return

    def close(self) -> None:
        """Stop accepting events and let :meth:`pump` finish."""
        self._closed = True
        while True:
            try:
                self._queue.put_nowait(None)
                return
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()


# ---------------------------------------------------------------------------
# Websocket endpoint
# ---------------------------------------------------------------------------
def _clear_active_run(task: asyncio.Task[Any] | None = None) -> None:
    """Release the single-run guard if it still refers to ``task``."""
    global _active_run
    if not isinstance(_active_run, dict):
        return
    if task is None or _active_run.get("task") is task:
        _active_run = None


def _launch_run(goal: str, orchestrator: Orchestrator, relay: _EventRelay) -> asyncio.Task[Any]:
    """Register a run as the active one and start it in the background."""
    global _active_run, _last_state

    task: asyncio.Task[Any] = asyncio.create_task(orchestrator.run(), name="furrow-run")
    _active_run = {
        "goal": goal,
        "running": True,
        "state": _orchestrator_state(orchestrator),
        "task": task,
    }

    def _on_done(finished: asyncio.Task[Any]) -> None:
        global _last_state
        _last_state = _result_state(finished)
        _clear_active_run(finished)
        error = None if finished.cancelled() else finished.exception()
        if error is not None:
            relay({"kind": "error", "cycle": 0, "message": f"run failed: {error}"})

    task.add_done_callback(_on_done)
    return task


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    relay = _EventRelay(websocket)
    sender = asyncio.create_task(relay.pump(), name="furrow-ws-sender")
    emit = _never_raises(relay)
    orchestrator: Orchestrator | None = None
    runner: asyncio.Task[Any] | None = None
    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except (WebSocketDisconnect, WebSocketDisconnected):
                break
            except (KeyError, RuntimeError) as exc:
                # Binary or otherwise unusable frame: report it and stop reading.
                emit({"kind": "error", "cycle": 0, "message": f"unsupported frame: {exc}"})
                break
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("payload must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                emit({"kind": "error", "cycle": 0, "message": f"invalid JSON payload: {exc}"})
                continue
            action = str(payload.get("action") or "").strip().lower()
            if action == "stop":
                if orchestrator is None:
                    emit({"kind": "error", "cycle": 0, "message": "no run in progress"})
                else:
                    orchestrator.request_stop()
                    emit({"kind": "stop_acknowledged", "cycle": 0})
                continue
            if action:
                emit(
                    {
                        "kind": "error",
                        "cycle": 0,
                        "message": f"unknown action: {action!r}",
                    }
                )
                continue
            try:
                request = StartRequest.model_validate(payload)
            except ValidationError as exc:
                emit(
                    {
                        "kind": "error",
                        "cycle": 0,
                        "message": f"invalid start payload: {exc.errors()}",
                    }
                )
                continue
            if not request.goal.strip():
                emit({"kind": "error", "cycle": 0, "message": "goal must not be empty"})
                continue
            if _active_run is not None:
                emit(
                    {
                        "kind": "error",
                        "cycle": 0,
                        "message": "a run is already in progress; stop it first",
                        "goal": _active_run.get("goal"),
                    }
                )
                continue
            # ``model`` is accepted for protocol compatibility; the orchestrator
            # picks its model up from the settings it was built with.
            orchestrator = _orchestrator_factory(goal=request.goal, on_event=emit)
            runner = _launch_run(request.goal, orchestrator, relay)
    except WebSocketDisconnect:  # pragma: no cover - defensive
        pass
    finally:
        if orchestrator is not None:
            with contextlib.suppress(Exception):
                orchestrator.request_stop()
        if runner is not None:
            # Never leave an orphaned run behind; cancel it if it refuses to stop.
            with contextlib.suppress(Exception, asyncio.TimeoutError):
                await asyncio.wait_for(runner, timeout=STOP_TIMEOUT)
            _clear_active_run(runner)
        relay.close()
        with contextlib.suppress(Exception):
            await sender


def run(host: str = "0.0.0.0", port: int = 8000, *, log_level: str | None = None) -> None:
    """Serve the web UI with uvicorn (blocking)."""
    if log_level:
        uvicorn.run(app, host=host, port=port, log_level=log_level)
    else:
        uvicorn.run(app, host=host, port=port)