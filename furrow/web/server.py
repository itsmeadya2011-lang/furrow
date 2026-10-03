"""Web UI for Furrow.

The browser opens a websocket to ``/ws`` and receives one JSON object per
orchestrator event, so a long run streams its progress instead of leaving the
page on "Starting...".

Protocol
--------
client -> server (text frames):

* ``{"goal": "do the thing", "model": "optional-model-id"}`` -- start a run.
* ``{"action": "stop"}`` -- ask the active run to wind down.

server -> client (text frames, one JSON object each):

* ``{"event": "<orchestrator event>", ...payload}`` -- e.g. ``cycle_start``,
  ``plan``, ``task_start``, ``task_done``, ``test_result``, ``cycle_end``,
  ``done`` and ``error`` as produced by ``Orchestrator``.
* ``{"event": "error", "code": ..., "message": ...}`` -- a problem with the
  client or the server state; the run did not (or could not) start.
* ``{"event": "stopping", "goal": ...}`` -- acknowledgement of a stop request.
* ``{"event": "run_finished", "status": "completed"|"stopped"|"failed",
  "goal": ..., "detail": ...}`` -- last message before the socket closes.

Any other client frame (non-JSON, non-object, unknown action) is answered with
an ``error`` message and the connection stays open.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")

#: How long the handler waits for a stopped run to unwind before cancelling it.
SHUTDOWN_TIMEOUT = 30.0

# Error codes used in ``{"event": "error", "code": ...}`` messages.
ERR_BAD_MESSAGE = "bad_message"
ERR_INVALID_PAYLOAD = "invalid_payload"
ERR_RUN_IN_PROGRESS = "run_in_progress"
ERR_RUN_FAILED = "run_failed"

#: Websocket close code used when a second run is refused.
CLOSE_RUN_IN_PROGRESS = 4001


class StartRequest(BaseModel):
    """Payload that starts a run."""

    goal: str
    model: str | None = None


@dataclass
class _ActiveRun:
    """The single run allowed to be in flight at any moment."""

    goal: str
    orchestrator: Any = None
    task: asyncio.Task[None] | None = None
    stop_requested: bool = False
    error: str = ""


@dataclass
class _EventChannel:
    """Bridges the orchestrator's synchronous ``on_event`` callback to the socket.

    Events are queued and drained by a single writer task, which keeps messages
    in order and means a vanished client can never break the run itself.
    """

    websocket: WebSocket
    queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    writer: asyncio.Task[None] | None = None
    closed: bool = False

    def start(self) -> None:
        self.writer = asyncio.create_task(self._drain())

    def emit(self, name: str, payload: dict[str, Any]) -> None:
        """``on_event`` callback handed to ``Orchestrator``; never raises."""
        if self.closed:
            return
        try:
            self.queue.put_nowait(_event_message(name, payload))
        except Exception:  # pragma: no cover - the queue is unbounded
            logger.debug("dropping %s event: queue unavailable", name, exc_info=True)

    async def send(self, message: dict[str, Any]) -> None:
        """Queue a server-generated message (errors, acknowledgements)."""
        if self.closed:
            return
        await self.queue.put(json.dumps(message, default=str))

    async def _drain(self) -> None:
        while True:
            message = await self.queue.get()
            if message is None:
                return
            try:
                await self.websocket.send_text(message)
            except Exception:
                # Client vanished mid-run: stop writing, keep the run alive.
                self.closed = True
                logger.debug("websocket send failed; event stream dropped", exc_info=True)
                return

    async def close(self) -> None:
        self.closed = True
        if self.writer is None:
            return
        with suppress(Exception):
            self.queue.put_nowait(None)
        writer, self.writer = self.writer, None
        with suppress(asyncio.CancelledError, Exception):
            await writer


# The registry is process-wide: two runs would fight over the same files.
_registry_lock = threading.Lock()
_active_run: _ActiveRun | None = None


def _claim_run(run: _ActiveRun) -> _ActiveRun | None:
    """Register ``run`` as the active run, or return the run already registered."""
    global _active_run
    with _registry_lock:
        if _active_run is not None:
            return _active_run
        _active_run = run
        return None


def _release_run(run: _ActiveRun) -> None:
    global _active_run
    with _registry_lock:
        if _active_run is run:
            _active_run = None


def _is_running() -> bool:
    with _registry_lock:
        return _active_run is not None


def _event_message(name: str, payload: dict[str, Any]) -> str:
    return json.dumps({"event": name, **payload}, default=str)


def _error_message(code: str, message: str) -> dict[str, Any]:
    return {"event": "error", "code": code, "message": message}


def _settings_for(model: str | None) -> Settings:
    """A fresh ``Settings`` per run -- the global singleton is never mutated."""
    if model is None:
        return Settings()
    return Settings(model=model)


async def _parse_start_request(raw: str, channel: _EventChannel) -> StartRequest | None:
    """Validate the first client frame; sends the error reply on failure."""
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        logger.debug("undecodable start payload: %s", exc)
        await _reject(channel, ERR_BAD_MESSAGE, f"Expected a JSON object, got: {raw[:120]!r}")
        return None
    if not isinstance(payload, dict):
        await _reject(channel, ERR_BAD_MESSAGE, "Expected a JSON object with a 'goal' key.")
        return None
    try:
        request = StartRequest.model_validate(payload)
    except ValidationError as exc:
        await _reject(channel, ERR_INVALID_PAYLOAD, _validation_message(exc))
        return None
    if not request.goal.strip():
        await _reject(channel, ERR_INVALID_PAYLOAD, "Goal must not be blank.")
        return None
    if request.model is not None and not request.model.strip():
        await _reject(channel, ERR_INVALID_PAYLOAD, "Model must not be blank when provided.")
        return None
    return request


def _validation_message(exc: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in error['loc']) or 'payload'}: {error['msg']}"
        for error in exc.errors()
    ]
    return "Invalid start payload -- " + "; ".join(problems)


async def _reject(channel: _EventChannel, code: str, message: str) -> None:
    await channel.send(_error_message(code, message))


def _request_stop(run: _ActiveRun) -> None:
    if run.orchestrator is None:
        return
    run.stop_requested = True
    try:
        run.orchestrator.stop()
    except Exception:
        logger.debug("orchestrator.stop() raised", exc_info=True)


async def _drive_run(run: _ActiveRun) -> None:
    try:
        await run.orchestrator.run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        run.error = str(exc) or exc.__class__.__name__
        logger.exception("orchestrator run failed")


async def _receive_loop(websocket: WebSocket, channel: _EventChannel, run: _ActiveRun) -> None:
    """Answer client control messages until the peer goes away."""
    while True:
        try:
            raw = await websocket.receive_text()
        except WebSocketDisconnect:
            return
        except Exception:
            logger.debug("websocket receive failed", exc_info=True)
            return
        try:
            message = json.loads(raw)
        except ValueError:
            await channel.send(
                _error_message(ERR_BAD_MESSAGE, f"Ignored non-JSON message: {raw[:120]!r}")
            )
            continue
        if not isinstance(message, dict):
            await channel.send(_error_message(ERR_BAD_MESSAGE, "Expected a JSON object."))
            continue
        action = message.get("action")
        if action == "stop":
            _request_stop(run)
            await channel.send({"event": "stopping", "goal": run.goal})
        else:
            await channel.send(
                _error_message(ERR_BAD_MESSAGE, f"Unsupported action: {action!r}")
            )


async def _finish_run(run: _ActiveRun) -> None:
    """Stop and reap the run task, then free the registry slot.

    The slot is released in a ``finally`` so a cancelled handler (client gone,
    server shutting down, ...) can never wedge the server permanently.
    """
    try:
        task = run.task
        if task is not None and not task.done():
            _request_stop(run)
            try:
                await asyncio.wait_for(task, timeout=SHUTDOWN_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("run did not stop within %.0fs; cancelling", SHUTDOWN_TIMEOUT)
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
    finally:
        _release_run(run)


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0 auto;
         max-width: 52rem; padding: 1.5rem; line-height: 1.45; }
  h1 { margin: 0 0 0.75rem; font-size: 1.4rem; }
  #form { display: flex; gap: 0.5rem; flex-wrap: wrap; }
  #goal { flex: 1 1 20rem; padding: 0.5rem; font: inherit; }
  button { padding: 0.5rem 0.9rem; font: inherit; cursor: pointer; }
  button[disabled] { opacity: 0.5; cursor: not-allowed; }
  #status { color: #666; font-size: 0.9rem; margin: 0.75rem 0; }
  #out { background: rgba(127, 127, 127, 0.08); border-radius: 6px;
         padding: 0.75rem; max-height: 65vh; overflow: auto;
         font-size: 0.8rem; margin: 0; }
  .line { border-bottom: 1px solid rgba(127, 127, 127, 0.2); padding: 0.35rem 0; }
  .head { font-weight: 600; }
  .body { margin: 0.2rem 0 0; white-space: pre-wrap; word-break: break-word; }
</style>
</head>
<body>
<h1>Furrow</h1>
<form id="form">
  <input id="goal" placeholder="Describe the goal..." required autocomplete="off" />
  <button id="start" type="submit">Start</button>
  <button id="stop" type="button" disabled>Stop</button>
</form>
<p id="status">Idle</p>
<pre id="out"></pre>
<script>
const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
const endpoint = scheme + '//' + location.host + '/ws';
const form = document.getElementById('form');
const goalInput = document.getElementById('goal');
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const statusEl = document.getElementById('status');
const out = document.getElementById('out');
let socket = null;

function setStatus(text) { statusEl.textContent = text; }

function setRunning(running) {
  startBtn.disabled = running;
  stopBtn.disabled = !running;
  goalInput.disabled = running;
}

// textContent escapes the payload, so event data can never inject markup.
function log(label, value) {
  const line = document.createElement('div');
  line.className = 'line';
  const head = document.createElement('span');
  head.className = 'head';
  head.textContent = '[' + new Date().toLocaleTimeString() + '] ' + label;
  const body = document.createElement('pre');
  body.className = 'body';
  body.textContent = value;
  line.appendChild(head);
  line.appendChild(body);
  out.appendChild(line);
  out.scrollTop = out.scrollHeight;
}

form.onsubmit = (event) => {
  event.preventDefault();
  const goal = goalInput.value.trim();
  if (!goal) { setStatus('Enter a goal first.'); return; }
  out.textContent = '';
  setRunning(true);
  setStatus('Connecting...');
  socket = new WebSocket(endpoint);
  socket.onmessage = (ev) => {
    let message;
    try { message = JSON.parse(ev.data); }
    catch (err) { log('unparsed', ev.data); return; }
    const name = message.event || 'message';
    const payload = Object.assign({}, message);
    delete payload.event;
    log(name, JSON.stringify(payload, null, 2));
    if (name === 'error') { setStatus('Error: ' + (message.message || 'unknown')); }
    else if (name === 'stopping') { setStatus('Stopping...'); }
    else if (name === 'task_start') { setStatus('Running: ' + (message.description || '')); }
    else if (name === 'run_finished') { setStatus('Finished: ' + (message.status || 'completed')); }
  };
  socket.onclose = () => {
    setRunning(false);
    socket = null;
    log('socket', 'closed');
  };
  socket.onerror = () => log('socket', 'connection error');
  socket.send(JSON.stringify({ goal: goal }));
};

stopBtn.onclick = () => {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    setStatus('Not connected.');
    return;
  }
  socket.send(JSON.stringify({ action: 'stop' }));
  setStatus('Stopping...');
};
</script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    channel = _EventChannel(websocket=websocket)
    run: _ActiveRun | None = None
    close_code = 1000
    try:
        channel.start()
        try:
            raw = await websocket.receive_text()
        except WebSocketDisconnect:
            return
        request = await _parse_start_request(raw, channel)
        if request is None:
            close_code = 1003
            return

        # Claim the slot before building an orchestrator, so a rejected start
        # never spins up a competitor.
        run = _ActiveRun(goal=request.goal.strip())
        active = _claim_run(run)
        if active is not None:
            await channel.send(
                _error_message(
                    ERR_RUN_IN_PROGRESS,
                    f"A run is already in progress ({active.goal!r}); "
                    "stop it or wait for it to finish before starting another.",
                )
            )
            close_code = CLOSE_RUN_IN_PROGRESS
            return

        run.orchestrator = Orchestrator(
            goal=run.goal,
            on_event=channel.emit,
            settings=_settings_for(request.model),
        )
        run.task = asyncio.create_task(_drive_run(run))
        receiver = asyncio.create_task(_receive_loop(websocket, channel, run))
        waitables: set[asyncio.Task[None]] = {receiver, run.task}
        try:
            await asyncio.wait(waitables, return_when=asyncio.FIRST_COMPLETED)
        finally:
            receiver.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await receiver

        await channel.send(
            {
                "event": "run_finished",
                "status": _run_status(run),
                "goal": run.goal,
                "detail": run.error,
            }
        )
    except WebSocketDisconnect:
        logger.debug("client disconnected from /ws")
    except Exception as exc:
        logger.exception("websocket handler failed")
        await channel.send(_error_message(ERR_RUN_FAILED, f"Run failed: {exc}"))
        close_code = 1011
    finally:
        if run is not None:
            await _finish_run(run)
        await channel.close()
        with suppress(Exception):
            await websocket.close(code=close_code)


def _run_status(run: _ActiveRun) -> str:
    if run.error:
        return "failed"
    if run.stop_requested:
        return "stopped"
    return "completed"


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Serve the UI with uvicorn (blocking)."""
    uvicorn.run(app, host=host, port=port)


def is_running() -> bool:
    """True while a run started through the websocket is still active."""
    return _is_running()
