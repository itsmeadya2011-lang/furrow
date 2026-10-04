"""Web UI for Furrow: one page plus a websocket that streams a run as JSON events.

Wire protocol for ``/ws``:

* the client sends ``{"goal": "...", "model": "..."}`` once, right after connecting;
* the server replies with one JSON message per orchestrator event, shaped
  ``{"type": <event>, "cycle": <int|null>, "message": <str>, ...}`` plus whatever
  extra keys that event carries (``passed``/``failures`` for ``test.result``, ...);
* the stream ends with the terminal ``run.done`` event, or with an ``error`` event
  when the goal was rejected before a run could start.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from furrow.config import Settings
from furrow.core import LoopResult, Orchestrator
from furrow.core.orchestrator import EventHandler
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)

# Loopback by default: this endpoint has no authentication and it drives an agent
# that executes subprocesses and writes files, so it must not be exposed to a
# network (or the internet) unless the operator asks for it explicitly.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
EMPTY_GOAL_MESSAGE = "A goal is required: describe what Furrow should build."

app = FastAPI(title="Furrow")

# asyncio only keeps weak references to pending tasks, so a background run would be
# collectable the moment this module dropped it. Every live run stays here until it
# is genuinely finished, which is what keeps a run alive across a client reconnect.
_ACTIVE_RUNS: set[asyncio.Task[LoopResult]] = set()


class StartRequest(BaseModel):
    """Validated start request for :func:`start_endpoint`."""

    goal: str = Field(min_length=1, description="What Furrow should accomplish.")
    model: str | None = Field(default=None, description="Override the model for every role.")


# --------------------------------------------------------------------------------------
# Run plumbing
# --------------------------------------------------------------------------------------


def _settings_for(model: str | None = None) -> Settings:
    """Build the :class:`Settings` used by a single run.

    The settings are constructed per run and never taken from (nor written to) the process-wide
    ``get_settings`` cache: that cache is *rebuilt* by ``get_settings(**overrides)``, so a single
    request carrying a model would otherwise become the server-wide default for every later run,
    and a run without a model would silently inherit whatever the previous request overrode.
    ``Settings`` is a ``BaseSettings`` with ``env_prefix="FURROW_"`` and ``env_file=".env"``, so
    building it here still layers this run's overrides on the ambient environment and ``.env``.
    """
    overrides: dict[str, Any] = {}
    name = (model or "").strip()
    if name:
        # A single model applies to every role, matching `furrow start --model`.
        overrides.update(model=name, planner_model=name, worker_model=name, tester_model=name)
    return Settings(**overrides)


async def _send(websocket: WebSocket, name: str, payload: dict[str, Any] | None = None) -> bool:
    """Send one JSON message to the browser; return ``False`` if the client is gone.

    The orchestrator awaits its event handler, so this sits on the hot path of the
    run. Every send is guarded: once the peer has gone, sending raises, and that
    must end the stream quietly instead of taking the run down or wedging it.
    """
    message: dict[str, Any] = {"type": name}
    for key, value in (payload or {}).items():
        if key != "type":  # the event name always wins over anything in the payload
            message[key] = value
    message.setdefault("message", "")
    message.setdefault("cycle", None)
    try:
        await websocket.send_json(message)
    except (WebSocketDisconnect, RuntimeError, OSError) as exc:
        # starlette raises WebSocketDisconnect on an abrupt close and
        # WebSocketDisconnected (a RuntimeError) once the socket is closed locally.
        logger.debug("Dropping %s event, client is gone: %s", name, exc)
        return False
    return True


def _stream_handler(websocket: WebSocket) -> EventHandler:
    """Build the ``on_event`` callback that forwards run progress to the browser."""

    async def on_event(name: str, payload: dict[str, Any]) -> None:
        await _send(websocket, name, payload)

    return on_event


async def _drive(orchestrator: Orchestrator, websocket: WebSocket) -> LoopResult:
    """Run the loop, turning an unexpected crash into a terminal ``error`` event."""
    try:
        return await orchestrator.run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # deliberate: tell the UI, log it, then keep it visible
        logger.exception("Furrow run crashed")
        await _send(websocket, "error", {"message": f"Furrow run crashed: {exc}"})
        raise


async def _watch_disconnect(websocket: WebSocket, closed: asyncio.Event) -> None:
    """Block until the peer disconnects, then set ``closed``.

    starlette surfaces a client-side close as a ``websocket.disconnect`` receive
    message. No further client messages are expected, so anything else is ignored.
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            closed.set()
            return


async def _await_stream_end(websocket: WebSocket, task: asyncio.Task[LoopResult]) -> bool:
    """Wait for the run to finish or for the client to leave; True if the client left.

    Returning from a websocket endpoint closes the connection, so the handler has to
    stay alive for the whole run. It waits on a disconnect watcher rather than
    awaiting the loop inline, which keeps stream teardown (and cancellation) here.
    """
    closed = asyncio.Event()
    watcher = asyncio.create_task(_watch_disconnect(websocket, closed), name="furrow-ws-watch")
    try:
        await asyncio.wait({watcher, task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # The watcher must be finished before anyone closes the socket: calling
        # receive() after close raises.
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watcher
    return closed.is_set()


async def _reap(task: asyncio.Task[LoopResult] | None) -> None:
    """Cancel the run if it is still going, and always wait for it to finish.

    ``Orchestrator.run`` catches ``CancelledError``, cancels its in-flight workers and
    returns a ``cancelled`` LoopResult, so the cancel lands inside the loop instead of
    leaving an agent that keeps spending API credits after the browser is gone.
    Awaiting here also retrieves any exception, so asyncio never reports "exception was
    never retrieved", and ``_ACTIVE_RUNS`` holds the reference until the task is truly
    done, so an early return cannot collect a cancelling task.
    """
    if task is None:
        return
    if not task.done():
        logger.info("Cancelling the Furrow run")
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def _close(websocket: WebSocket) -> None:
    """Close the socket politely now that the stream is over."""
    with contextlib.suppress(WebSocketDisconnect, RuntimeError, OSError):
        await websocket.close()


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Stream one Furrow run to the browser over a websocket."""
    await websocket.accept()
    task: asyncio.Task[LoopResult] | None = None
    try:
        try:
            data = await websocket.receive_json()
        except (TypeError, ValueError) as exc:  # receive_json raises on non-JSON input
            logger.debug("Rejected a malformed start payload: %s", exc)
            await _send(websocket, "error", {"message": f"Malformed start request: {exc}"})
            return
        if not isinstance(data, dict):
            await _send(websocket, "error", {"message": "Expected a JSON object with a goal."})
            return
        goal = str(data.get("goal") or "").strip()
        if not goal:
            await _send(websocket, "error", {"message": EMPTY_GOAL_MESSAGE})
            return
        raw_model = data.get("model")
        model = raw_model.strip() if isinstance(raw_model, str) else None
        settings = _settings_for(model)
        orchestrator = Orchestrator(
            goal=goal,
            client=LLMClient(settings=settings),
            on_event=_stream_handler(websocket),
        )
        task = asyncio.create_task(_drive(orchestrator, websocket), name="furrow-run")
        _ACTIVE_RUNS.add(task)
        task.add_done_callback(_ACTIVE_RUNS.discard)
        if not await _await_stream_end(websocket, task):
            await _close(websocket)
    except WebSocketDisconnect:
        logger.info("Websocket client disconnected; stopping the run")
    finally:
        await _reap(task)


@app.post("/api/start")
async def start_endpoint(request: StartRequest) -> dict[str, Any]:
    """Validate a run request and report the configuration it would use.

    The run itself streams over ``/ws``; this endpoint exists so a browser, a script
    or a human can check a goal and see the effective settings before opening a
    socket. It never starts a run, which is why it stays safe to call.
    """
    goal = request.goal.strip()
    if not goal:
        raise HTTPException(status_code=422, detail=EMPTY_GOAL_MESSAGE)
    settings = _settings_for(request.model)
    return {
        "goal": goal,
        "provider": getattr(settings.provider, "value", str(settings.provider)),
        "model": settings.model,
        "planner_model": settings.planner_model,
        "worker_model": settings.worker_model,
        "tester_model": settings.tester_model,
        "workspace": str(settings.workspace),
        "max_cycles": settings.max_cycles,
        "max_parallel_tasks": settings.max_parallel_tasks,
        "websocket": "/ws",
    }


_INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; margin: 0 auto; max-width: 54rem; padding: 1.5rem; }
  h1 { margin: 0 0 .25rem; }
  .meta { color: #6b7280; font-size: .9rem; }
  #form { display: flex; gap: .5rem; margin: 1rem 0 .5rem; }
  #goal { flex: 1; padding: .5rem; font: inherit; }
  button { padding: .5rem 1.25rem; font: inherit; }
  button[disabled] { opacity: .55; cursor: progress; }
  #status { font-weight: 600; margin: .5rem 0; }
  #log {
    background: rgba(127, 127, 127, .14); border-radius: .5rem; padding: .75rem;
    max-height: 60vh; overflow: auto; white-space: pre-wrap;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85rem;
  }
  .pass { color: #15803d; }
  .fail { color: #b91c1c; }
  .warn { color: #b45309; }
  .meta { color: #6b7280; }
</style>
</head>
<body>
<h1>Furrow</h1>
<p class="meta">Autonomous coding agent: plans, executes and tests until the goal is met.</p>
<form id="form">
  <input id="goal" placeholder="Describe the goal for Furrow" autocomplete="off" required />
  <button id="submit" type="submit">Start</button>
</form>
<p id="status">Idle.</p>
<div id="log"></div>
<script>
  const form = document.getElementById('form');
  const goalInput = document.getElementById('goal');
  const submit = document.getElementById('submit');
  const status = document.getElementById('status');
  const log = document.getElementById('log');
  let cycle = null;
  let finished = false;

  // Every model-, filesystem- or error-derived string is written with textContent,
  // never innerHTML, so markup in the payload is displayed instead of executed.
  function line(text, kind) {
    const row = document.createElement('div');
    if (kind) { row.className = kind; }
    row.textContent = text;
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;
  }

  function setStatus(text) { status.textContent = text; }

  function setBusy(busy) {
    submit.disabled = busy;
    goalInput.disabled = busy;
    submit.textContent = busy ? 'Running' : 'Start';
  }

  function socketUrl() {
    // https pages need wss://; building the scheme from location keeps that working.
    const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return scheme + '//' + location.host + '/ws';
  }

  function render(data) {
    if (Number.isInteger(data.cycle)) { cycle = data.cycle; }
    const prefix = cycle === null ? '' : 'cycle ' + cycle + ': ';
    const message = data.message || data.type || 'event';
    const type = data.type || 'event';
    if (type === 'test.result') {
      const passed = data.passed === true;
      line(prefix + 'tests ' + (passed ? 'passed' : 'failed') + ': ' + (data.summary || ''),
           passed ? 'pass' : 'fail');
      (data.failures || []).forEach((failure) => line('  - ' + failure, 'fail'));
    } else if (type === 'run.done') {
      finished = true;
      line(prefix + message, data.reason === 'goal-complete' ? 'pass' : 'warn');
      setStatus('Finished after ' + (data.cycles || 0) + ' cycles: ' + (data.reason || 'unknown'));
    } else if (type === 'error') {
      finished = true;
      line(prefix + message, 'fail');
      setStatus(message);
    } else if (type === 'task.done') {
      const files = data.files || [];
      line(prefix + message + (files.length ? ' [' + files.join(', ') + ']' : ''), 'pass');
    } else if (type === 'task.failed' || type === 'task.skipped') {
      line(prefix + message, 'warn');
    } else {
      line(prefix + message, type === 'cycle.start' ? 'meta' : '');
    }
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const goal = goalInput.value.trim();
    if (!goal) {
      setStatus('Enter a goal first.');
      goalInput.focus();
      return;
    }
    log.replaceChildren();
    cycle = null;
    finished = false;
    setBusy(true);
    setStatus('Starting...');
    line('Goal: ' + goal, 'meta');
    const socket = new WebSocket(socketUrl());
    socket.onopen = () => socket.send(JSON.stringify({ goal: goal }));
    socket.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (err) {
        line('Unreadable message from the server.', 'warn');
        return;
      }
      render(data);
    };
    socket.onerror = () => setStatus('Could not reach the Furrow server.');
    socket.onclose = () => {
      setBusy(false);
      if (!finished) { setStatus('Connection closed before the run finished.'); }
    };
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page UI."""
    return HTMLResponse(content=_INDEX_HTML)


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, log_level: str | None = None) -> None:
    """Serve the web UI with uvicorn.

    Binding a non-loopback address publishes an unauthenticated endpoint that runs
    an LLM agent, executes subprocesses and writes files, so anything other than
    ``127.0.0.1`` is a deliberate, logged decision by the operator.
    """
    if host not in LOOPBACK_HOSTS:
        logger.warning(
            "Furrow is serving on %s without authentication; anyone who can reach it "
            "can run an agent that executes commands and writes files.",
            host,
        )
    uvicorn.run(app, host=host, port=port, log_level=(log_level or "info").lower())
