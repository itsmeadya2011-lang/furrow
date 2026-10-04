from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from furrow import __version__
from furrow.config import get_settings
from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

MAX_GOAL_CHARS = 4000
RUN_SHUTDOWN_TIMEOUT = 10.0
WS_CLOSE_INVALID_PAYLOAD = 1003
WS_CLOSE_DENIED = 1008
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    """The single payload a browser sends on `/ws` to open a run."""

    goal: str = Field(min_length=1, max_length=MAX_GOAL_CHARS)

    @field_validator("goal")
    @classmethod
    def _goal_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("goal must not be blank")
        return value.strip()


class _InvalidRequest(ValueError):
    """The client sent something that is not a usable start request."""


@dataclass
class _ActiveRun:
    goal: str
    started_at: float


# One run at a time: N tabs would mean N orchestrators hammering a single API key.
_active_run: _ActiveRun | None = None


def _claim_run(goal: str) -> bool:
    global _active_run
    if _active_run is not None:
        return False
    _active_run = _ActiveRun(goal=goal, started_at=time.time())
    return True


def _release_run() -> None:
    global _active_run
    _active_run = None


def _origin_allowed(origin: str | None) -> bool:
    """Guard against cross-site WebSocket hijacking.

    CORS does not apply to WebSocket handshakes, so without this check any page the
    user visits could open `ws://127.0.0.1:8000/ws` and make the server act with the
    server's credentials. A missing Origin means a non-browser client (CLI, tests),
    which cannot be a hijacking vector.
    """
    if not origin:
        return True
    parsed = urlsplit(origin)
    if parsed.scheme not in ("http", "https"):
        return False
    return (parsed.hostname or "").lower() in LOOPBACK_HOSTS


def _make_sink(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[dict[str, Any] | None],
) -> Callable[[dict[str, Any] | None], None]:
    """Build the synchronous `on_event` callback handed to the orchestrator.

    `on_event` is called from the orchestrator's own code and must never block or
    raise, so it only schedules a non-blocking `put_nowait` on the loop that owns the
    queue. `call_soon_threadsafe` is safe whether the callback fires on this loop or
    on another thread, and keeps event order. Every item, including the end-of-run
    sentinel, goes through this one path: putting the sentinel straight onto the queue
    would jump ahead of events that are still only scheduled, and the pump would stop
    before forwarding them.
    """

    def push(event: dict[str, Any] | None) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except RuntimeError:  # loop already closed: the run is being torn down
            logger.debug("dropping event %r: event loop is closed", event)

    return push


async def _drive(
    orchestrator: Orchestrator,
    push: Callable[[dict[str, Any] | None], None],
) -> None:
    """Run the orchestrator, reporting failures as an event and always ending the pump."""
    try:
        await orchestrator.run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # a traceback must never reach the browser
        logger.exception("run failed")
        push({"type": "error", "message": f"Run failed: {exc}"})
    finally:
        push(None)


async def _pump(websocket: WebSocket, queue: asyncio.Queue[Any]) -> None:
    """Forward queued events to the browser until the sentinel or a closed socket."""
    try:
        while True:
            event = await queue.get()
            if event is None:
                return
            await websocket.send_json(event)
    except (WebSocketDisconnect, RuntimeError):
        return


async def _watch_disconnect(websocket: WebSocket) -> None:
    """Resolve as soon as the client goes away, so the run can be cancelled."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


async def _shutdown(
    orchestrator: Orchestrator,
    run_task: asyncio.Task[None],
    helpers: list[asyncio.Task[None]],
) -> None:
    """Stop the run cooperatively, then force anything left, then drop the helpers."""
    if not run_task.done():
        orchestrator.cancel()
    with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(run_task), RUN_SHUTDOWN_TIMEOUT)
    if not run_task.done():
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await run_task
    for helper in helpers:
        helper.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await helper


def _describe(exc: ValidationError) -> str:
    problems = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"]) or "payload"
        detail = str(error["msg"]).removeprefix("Value error, ")
        problems.append(f"{where}: {detail}")
    return "Invalid start request (" + "; ".join(problems) + ")"


async def _read_start_request(websocket: WebSocket) -> StartRequest:
    try:
        raw = await websocket.receive_text()
    except AssertionError:
        raise _InvalidRequest("Start request must be a text frame.") from None
    try:
        payload = json.loads(raw)
    except ValueError:
        raise _InvalidRequest("Start request must be valid JSON.") from None
    if not isinstance(payload, dict):
        raise _InvalidRequest("Start request must be a JSON object.")
    try:
        return StartRequest.model_validate(payload)
    except ValidationError as exc:
        raise _InvalidRequest(_describe(exc)) from None


async def _reject(websocket: WebSocket, message: str, code: int) -> None:
    try:
        await websocket.send_json({"type": "error", "message": message})
    except (WebSocketDisconnect, RuntimeError):
        return
    with contextlib.suppress(WebSocketDisconnect, RuntimeError):
        await websocket.close(code=code, reason=message[:120])


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Furrow</title>
<style>
body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 2rem; }
h1 { margin-bottom: 0.25rem; }
label { display: block; margin-bottom: 0.25rem; font-weight: 600; }
input { width: min(60rem, 100%); padding: 0.5rem; }
button { padding: 0.5rem 1rem; margin-right: 0.5rem; }
#out { border: 1px solid #8884; border-radius: 0.25rem; padding: 0.75rem;
       min-height: 12rem; white-space: pre-wrap; margin-top: 1rem; }
.error { color: #c0392b; }
.warn { color: #b9770e; }
.ok { color: #1e8449; }
</style>
</head>
<body>
<h1>Furrow</h1>
<form id="form">
  <label for="goal">Goal</label>
  <input id="goal" name="goal" placeholder="Enter goal" autocomplete="off" required>
  <button id="start" type="submit">Start</button>
  <button id="stop" type="button" disabled>Cancel</button>
</form>
<pre id="out"></pre>
<script>
const NL = String.fromCharCode(10);
const form = document.getElementById('form');
const out = document.getElementById('out');
const goalInput = document.getElementById('goal');
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
let socket = null;

function append(text, kind) {
  const line = document.createElement('span');
  line.className = kind ? 'line ' + kind : 'line';
  line.textContent = text + NL;
  out.appendChild(line);
  out.scrollTop = out.scrollHeight;
}

function wsUrl() {
  const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
  return scheme + location.host + '/ws';
}

function setRunning(running) {
  startBtn.disabled = running;
  stopBtn.disabled = !running;
  goalInput.disabled = running;
}

form.onsubmit = (ev) => {
  ev.preventDefault();
  if (socket) {
    append('A run is already connected from this tab.', 'warn');
    return;
  }
  const goal = goalInput.value.trim();
  if (!goal) {
    append('Enter a goal first.', 'error');
    return;
  }
  append('Starting: ' + goal);
  setRunning(true);
  const ws = new WebSocket(wsUrl());
  socket = ws;
  ws.onopen = () => ws.send(JSON.stringify({goal: goal}));
  ws.onmessage = (ev) => {
    let payload = null;
    try {
      payload = JSON.parse(ev.data);
    } catch (err) {
      append(String(ev.data));
      return;
    }
    if (payload && typeof payload.message === 'string') {
      append(payload.message, payload.type);
    } else {
      append(JSON.stringify(payload));
    }
  };
  ws.onerror = () => {
    append('WebSocket error - is the server still running? Check its console.', 'error');
  };
  ws.onclose = (ev) => {
    socket = null;
    setRunning(false);
    append('Closed (' + ev.code + (ev.reason ? ': ' + ev.reason : '') + ').');
  };
};

stopBtn.onclick = () => {
  if (socket) {
    append('Cancelling run...', 'warn');
    socket.close(1000, 'client cancel');
  }
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe: cheap, and it never leaks the goal of the active run."""
    return {"status": "ok", "version": __version__, "active_run": _active_run is not None}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    origin = websocket.headers.get("origin")
    if not _origin_allowed(origin):
        logger.warning("rejected websocket handshake from origin %r", origin)
        await websocket.close(code=WS_CLOSE_DENIED, reason="origin not allowed")
        return

    await websocket.accept()
    orchestrator: Orchestrator | None = None
    run_task: asyncio.Task[None] | None = None
    helpers: list[asyncio.Task[None]] = []
    claimed = False
    try:
        try:
            request = await _read_start_request(websocket)
        except WebSocketDisconnect:
            return
        except _InvalidRequest as exc:
            await _reject(websocket, str(exc), WS_CLOSE_INVALID_PAYLOAD)
            return

        if not _claim_run(request.goal):
            await _reject(
                websocket,
                "A run is already in progress; close the other tab first.",
                WS_CLOSE_DENIED,
            )
            return
        claimed = True

        queue: asyncio.Queue[Any] = asyncio.Queue()
        push = _make_sink(asyncio.get_running_loop(), queue)
        try:
            orchestrator = Orchestrator(goal=request.goal, on_event=push)
        except Exception as exc:  # e.g. no API key configured: say so, do not traceback
            logger.exception("could not build an orchestrator")
            await _reject(websocket, f"Could not start the run: {exc}", WS_CLOSE_DENIED)
            return
        run_task = asyncio.create_task(_drive(orchestrator, push), name="furrow-run")
        pump = asyncio.create_task(_pump(websocket, queue), name="furrow-events")
        watcher = asyncio.create_task(_watch_disconnect(websocket), name="furrow-watch")
        helpers = [pump, watcher]
        await asyncio.wait({pump, watcher}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        if orchestrator is not None and run_task is not None:
            await _shutdown(orchestrator, run_task, helpers)
        if claimed:
            _release_run()
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await websocket.close()


def run(host: str | None = None, port: int | None = None) -> None:
    """Serve the web UI.

    Defaults come from `settings.web_host` / `settings.web_port`, which bind loopback
    (`127.0.0.1:8000`). There is no authentication: this server writes files and makes
    billable LLM calls, so anything other than loopback is an explicit, warned choice.
    """
    settings = get_settings()
    resolved_host = settings.web_host if host is None else host
    resolved_port = settings.web_port if port is None else port
    if not 1 <= resolved_port <= 65535:
        raise ValueError(f"port must be between 1 and 65535, got {resolved_port}")
    if resolved_host not in LOOPBACK_HOSTS:
        logger.warning(
            "binding %s:%s exposes an unauthenticated agent that writes files and "
            "spends API credits to everything that can reach it",
            resolved_host,
            resolved_port,
        )
    uvicorn.run(app, host=resolved_host, port=resolved_port)
