"""Browser front-end for the orchestrator loop.

Wire protocol
-------------
One websocket carries exactly one run.  Every frame is a JSON object with the
same two keys, ``type`` and ``message``; the bundled JavaScript parses exactly
those two and nothing else:

==========================================  ====================================
frame (server -> client)                    meaning
==========================================  ====================================
``{"type": "started", "message": "..."}``   run accepted; the message carries
                                            the goal and the cycle budget
``{"type": "event",   "message": "..."}``   one single-line orchestrator event
``{"type": "done",    "message": "..."}``   ``run()`` returned on its own; the
                                            message is the ``stop_reason``
``{"type": "stopped", "message": "..."}``   cancelled after a ``stop`` request
``{"type": "error",   "message": "..."}``   the orchestrator raised; the run is
                                            over
==========================================  ====================================

The client sends two frames only::

    {"goal": "add a health check", "max_cycles": 3}   # start (first frame)
    {"type": "stop"}                                  # cancel the run

The server closes the socket right after the terminal frame (``done``,
``stopped`` or ``error``), so a client can always tell that the run is over.
A client that disappears mid-run gets no terminal frame -- there is nobody left
to receive it -- but the loop is cancelled either way.

Cycle budget
------------
``Orchestrator.run()`` loops forever by design, so the web path is *always*
bounded: a session gets at most ``max_cycles`` cycles, taken from the client
frame when present (clamped into ``1..MAX_CYCLES_LIMIT``, so a requested ``0``
means "one cycle", never "endless"), otherwise ``settings.max_cycles`` when that
is non-zero, otherwise :data:`DEFAULT_MAX_CYCLES`.  The budget is handed to the
orchestrator as a per-session ``Settings`` copy; the global
``furrow.config.settings`` singleton is never mutated, so two browser sessions
cannot change each other's budget.

Security
--------
``run()`` binds ``127.0.0.1`` on purpose.  This server hands an autonomous,
code-writing agent to every client that connects and has no authentication of
any kind, so any other bind address publishes it to that network.  Only pass a
different ``host`` behind a trusted tunnel or firewall.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from furrow.config import Settings
from furrow.config import settings as default_settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Message types of the JSON envelope; the bundled JS switches on these strings.
TYPE_STARTED = "started"
TYPE_EVENT = "event"
TYPE_DONE = "done"
TYPE_STOPPED = "stopped"
TYPE_ERROR = "error"
TERMINAL_TYPES = frozenset({TYPE_DONE, TYPE_STOPPED, TYPE_ERROR})

# The only inbound message type besides the start frame.
STOP_TYPE = "stop"

# An endless loop behind a websocket is a resource leak, so a session without an
# explicit budget runs at most this many cycles (see the module docstring).
DEFAULT_MAX_CYCLES = 3
# Upper bound on a client-supplied budget: the browser may ask for more than the
# default, but not for "forever".
MAX_CYCLES_LIMIT = 50

# How long the sender may take to flush the terminal frame before the socket is
# torn down.  A client that stopped reading must not hold the run open.
DRAIN_TIMEOUT = 5.0

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

Message = dict[str, str]


class StartRequest(BaseModel):
    """The first client frame: what to work on and how many cycles it may use.

    ``max_cycles`` is optional and untrusted; :func:`_cycle_budget` clamps it.
    """

    goal: str
    max_cycles: int | None = None


def _one_line(value: object, limit: int = 500) -> str:
    """Frames are a single line of plain text, whatever produced them."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _message(kind: str, message: object) -> Message:
    """Build the one and only frame shape: exactly ``type`` and ``message``."""
    return {"type": kind, "message": _one_line(message)}


def _cycle_budget(requested: int | None) -> int:
    """Cycles this session may run: the client's request, else the config, else the default."""
    if requested is None:
        configured = default_settings.max_cycles
        return configured if configured > 0 else DEFAULT_MAX_CYCLES
    return min(max(requested, 1), MAX_CYCLES_LIMIT)


def _session_settings(max_cycles: int) -> Settings:
    """A per-session copy of the settings; the global singleton stays untouched."""
    return default_settings.model_copy(update={"max_cycles": max_cycles})


async def _send(websocket: WebSocket, frame: Message) -> bool:
    """Send one frame, reporting failure instead of raising: a dead socket is not an error."""
    try:
        await websocket.send_json(frame)
        return True
    except Exception:
        return False


async def _close(websocket: WebSocket) -> None:
    """Close the socket, tolerating a client that already went away."""
    with contextlib.suppress(Exception):
        await websocket.close()


async def _receive_json(websocket: WebSocket) -> Any:
    """``receive_json`` that reports a gone socket as :class:`WebSocketDisconnect`.

    Starlette raises a ``RuntimeError`` (``WebSocketDisconnected``) when a
    receive is attempted after the disconnect frame was consumed, so without
    this a disconnect during a run would look like an unrelated crash.
    """
    try:
        return await websocket.receive_json()
    except RuntimeError as exc:
        raise WebSocketDisconnect(code=1000) from exc


async def _read_start(websocket: WebSocket) -> StartRequest:
    """Parse the first client frame; raise ``ValueError`` with a usable reason."""
    frame = await _receive_json(websocket)
    try:
        start = StartRequest.model_validate(frame)
    except ValidationError as exc:
        raise ValueError(f"invalid start message: {exc}") from exc
    goal = start.goal.strip()
    if not goal:
        raise ValueError("start message needs a non-empty 'goal'")
    return start.model_copy(update={"goal": goal})


async def _pump(websocket: WebSocket, frames: asyncio.Queue[Message | None]) -> None:
    """Drain queued frames to the client until the ``None`` sentinel arrives.

    The queue is unbounded because ``on_event`` is a synchronous callback that
    cannot await back-pressure; the per-session cycle budget bounds the total.
    """
    while True:
        frame = await frames.get()
        if frame is None:
            return
        if not await _send(websocket, frame):
            return


async def _watch_for_stop(websocket: WebSocket) -> None:
    """Wait for an inbound ``{"type": "stop"}``; raise when the socket is gone.

    Unknown inbound frames are ignored so a newer client cannot break a run.
    """
    while True:
        try:
            frame = await _receive_json(websocket)
        except ValueError:
            continue  # unparsable inbound frame: not worth failing the run over
        if isinstance(frame, dict) and frame.get("type") == STOP_TYPE:
            return


async def _drive(orchestrator: Orchestrator) -> Message:
    """Run the loop to completion and map its outcome onto a single terminal frame."""
    try:
        await orchestrator.run()
    except asyncio.CancelledError:
        raise  # a stop request is a cancellation, not a failure: the caller reports "stopped"
    except Exception as exc:
        return _message(TYPE_ERROR, f"{type(exc).__name__}: {exc}")
    return _message(TYPE_DONE, orchestrator.stop_reason or "run finished")


async def _settle(*tasks: asyncio.Task[Any] | None) -> None:
    """Await cancelled tasks so no orchestrator task outlives the socket."""
    for task in tasks:
        if task is None:
            continue
        # Awaiting a cancelled task re-raises its CancelledError in this task,
        # and so does this task's own cancellation: both are expected here.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def _run_session(websocket: WebSocket, start: StartRequest) -> None:
    """Stream one orchestrator run to the browser and always clean it up."""
    budget = _cycle_budget(start.max_cycles)
    frames: asyncio.Queue[Message | None] = asyncio.Queue()
    run_task: asyncio.Task[Message] | None = None
    pump_task: asyncio.Task[None] | None = None
    watch_task: asyncio.Task[None] | None = None

    def on_event(line: str) -> None:
        frames.put_nowait(_message(TYPE_EVENT, line))

    try:
        started = _message(TYPE_STARTED, f"goal: {start.goal} | max_cycles: {budget}")
        if not await _send(websocket, started):
            return
        try:
            orchestrator = Orchestrator(
                goal=start.goal,
                on_event=on_event,
                settings=_session_settings(budget),
            )
        except Exception as exc:
            await _send(websocket, _message(TYPE_ERROR, f"{type(exc).__name__}: {exc}"))
            return

        run_task = asyncio.create_task(_drive(orchestrator), name="furrow-web-run")
        pump_task = asyncio.create_task(_pump(websocket, frames), name="furrow-web-sender")
        watch_task = asyncio.create_task(_watch_for_stop(websocket), name="furrow-web-stop-watch")

        finished, _ = await asyncio.wait({run_task, watch_task}, return_when=asyncio.FIRST_COMPLETED)
        if run_task in finished:
            terminal: Message | None = run_task.result()
        elif watch_task is not None and watch_task.exception() is not None:
            # The client vanished; nobody is left to receive a terminal frame.
            terminal = None
        else:
            terminal = _message(TYPE_STOPPED, "stopped on client request")

        if terminal is not None:
            frames.put_nowait(terminal)
            frames.put_nowait(None)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(pump_task, timeout=DRAIN_TIMEOUT)
    finally:
        # Whatever happened, the loop and the helpers stop before the socket does.
        for task in (watch_task, pump_task, run_task):
            if task is not None:
                task.cancel()
        await _settle(watch_task, pump_task, run_task)
        await _close(websocket)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Furrow</title>
<style>
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 2rem; max-width: 64rem; }
  input, button { font: inherit; padding: .25rem .5rem; }
  #out { border: 1px solid #ccc; padding: .75rem; height: 60vh; overflow: auto; }
  .meta { color: #666; }
  .error { color: #b00; }
  .done { color: #070; }
  .running { color: #06c; }
</style>
</head>
<body>
<h1>Furrow</h1>
<p>
  A run is capped at <strong>{{DEFAULT_MAX_CYCLES}}</strong> cycle(s) by default (override it below)
  and at {{MAX_CYCLES_LIMIT}} at most: the development loop is otherwise unbounded, so press
  <em>Stop</em> to end it early. This server has no authentication &mdash; keep it on 127.0.0.1.
</p>
<form id="form">
  <input id="goal" placeholder="Enter goal" required autocomplete="off" size="40" />
  <label for="max-cycles">max cycles</label>
  <input id="max-cycles" type="number" min="1" max="{{MAX_CYCLES_LIMIT}}" value="{{DEFAULT_MAX_CYCLES}}" size="4" />
  <button id="start" type="submit">Start</button>
  <button id="stop" type="button" disabled>Stop</button>
</form>
<p>State: <span id="state">idle</span></p>
<pre id="out"></pre>
<script>
  // Frames are {"type": ..., "message": ...}; type is one of started/event/done/stopped/error.
  var form = document.getElementById('form');
  var goalInput = document.getElementById('goal');
  var cyclesInput = document.getElementById('max-cycles');
  var startBtn = document.getElementById('start');
  var stopBtn = document.getElementById('stop');
  var out = document.getElementById('out');
  var stateEl = document.getElementById('state');
  var socket = null;
  var settled = false;

  function line(text, cls) {
    var row = document.createElement('div');
    row.textContent = text;
    if (cls) { row.className = cls; }
    out.appendChild(row);
    out.scrollTop = out.scrollHeight;
  }

  function setState(text, cls) {
    stateEl.textContent = text;
    stateEl.className = cls || '';
  }

  function setRunning(running) {
    startBtn.disabled = running;
    stopBtn.disabled = !running;
    goalInput.disabled = running;
    cyclesInput.disabled = running;
  }

  function socketUrl() {
    return (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
  }

  function handle(frame) {
    var kind = frame && frame.type;
    var text = (frame && frame.message) || '';
    if (kind === 'started') {
      line('> ' + text, 'meta');
    } else if (kind === 'event') {
      line(text);
    } else if (kind === 'done') {
      settled = true;
      line('Finished: ' + text, 'meta');
      setState('Finished: ' + text, 'done');
      setRunning(false);
    } else if (kind === 'stopped') {
      settled = true;
      line('Stopped: ' + text, 'meta');
      setState('Stopped: ' + text, 'meta');
      setRunning(false);
    } else if (kind === 'error') {
      settled = true;
      line('Error: ' + text, 'error');
      setState('Error: ' + text, 'error');
      setRunning(false);
    } else {
      line('Unrecognised frame: ' + JSON.stringify(frame), 'error');
    }
  }

  form.onsubmit = function (event) {
    event.preventDefault();
    out.textContent = '';
    settled = false;
    setRunning(true);
    setState('Running', 'running');
    socket = new WebSocket(socketUrl());
    socket.onopen = function () {
      socket.send(JSON.stringify({
        goal: goalInput.value,
        max_cycles: Number(cyclesInput.value) || null
      }));
    };
    socket.onmessage = function (event) { handle(JSON.parse(event.data)); };
    socket.onerror = function () {
      settled = true;
      setState('Error: the connection failed', 'error');
      setRunning(false);
    };
    socket.onclose = function () {
      if (!settled) {
        setState('Disconnected before the run finished', 'error');
        setRunning(false);
      }
      socket = null;
    };
  };

  stopBtn.onclick = function () {
    if (socket && socket.readyState === WebSocket.OPEN) {
      line('Stop requested', 'meta');
      socket.send(JSON.stringify({type: 'stop'}));
    }
  };
</script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    """The single-page UI: goal input, cycle budget, live log and a Stop button."""
    page = PAGE.replace("{{DEFAULT_MAX_CYCLES}}", str(DEFAULT_MAX_CYCLES))
    page = page.replace("{{MAX_CYCLES_LIMIT}}", str(MAX_CYCLES_LIMIT))
    return HTMLResponse(content=page)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Run one orchestrator per socket, streaming JSON frames until it ends."""
    await websocket.accept()
    try:
        start = await _read_start(websocket)
    except WebSocketDisconnect:
        await _close(websocket)
        return
    except ValueError as exc:
        await _send(websocket, _message(TYPE_ERROR, exc))
        await _close(websocket)
        return
    await _run_session(websocket, start)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the UI with uvicorn.

    Defaults to loopback on purpose: an autonomous code-writing agent is exposed
    to anyone who can reach the socket, and this server has no authentication.
    Binding another interface publishes it to that network, so only do it behind
    a trusted tunnel or firewall.
    """
    if host not in LOOPBACK_HOSTS:
        print(
            f"WARNING: serving Furrow on {host}:{port} exposes an autonomous, "
            "unauthenticated code-writing agent to that network.",
            file=sys.stderr,
        )
    uvicorn.run(app, host=host, port=port)