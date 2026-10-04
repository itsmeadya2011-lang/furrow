"""Web front-end: one websocket session streams a live agent run.

The orchestrator emits events through a *synchronous* :data:`EventSink`, but a
websocket can only be written with ``await``.  :class:`EventStream` decouples the
two halves: the sink only ever does ``queue.put_nowait`` (it never blocks and
never raises when the client has gone away) while a pump task drains the queue
with ``await websocket.send_json(...)``.  The buffer is bounded, so a slow or
dead client cannot stall the agent loop -- the oldest queued events are dropped
instead and the count is reported in the final ``session_end`` payload.

The loop is unbounded by default (``Settings.max_cycles == 0``), so every exit
path is explicit: the run halts by itself, the client sends
``{"action": "stop"}``, or the client disconnects.  In the latter two cases the
orchestrator task is cancelled and awaited, so no agent keeps burning tokens
after the socket is gone.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError

from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger("furrow.web")

EventPayload = dict[str, Any]
EventSink = Callable[[str, EventPayload], None]
OrchestratorFactory = Callable[[str, EventSink], Orchestrator]

#: Upper bound on buffered events.  Bounded so an unresponsive client can never
#: grow the queue without limit.
QUEUE_SIZE = 512
#: How long the sender waits for the socket to flush the last events.
DRAIN_TIMEOUT = 5.0
#: ``action`` values that stop a run.
STOP_ACTIONS = frozenset({"stop", "cancel", "abort", "halt", "quit"})

RUNNER_TASK = "furrow-orchestrator"
PUMP_TASK = "furrow-event-pump"
WATCHER_TASK = "furrow-client-watcher"
_TASK_NAMES = frozenset({RUNNER_TASK, PUMP_TASK, WATCHER_TASK})


class WebSocketLike(Protocol):
    """The slice of :class:`fastapi.WebSocket` that a session actually uses."""

    async def send_json(self, data: Any) -> None: ...

    async def receive_json(self) -> Any: ...


class StartRequest(BaseModel):
    """The only message the client sends to begin a run."""

    goal: str = Field(min_length=1)


app = FastAPI(title="Furrow")


def pending_session_tasks() -> list[asyncio.Task[Any]]:
    """Session tasks still alive; a leak check for tests."""
    return [
        task
        for task in asyncio.all_tasks()
        if task.get_name() in _TASK_NAMES and not task.done()
    ]


def _default_orchestrator(goal: str, on_event: EventSink) -> Orchestrator:
    """Build the real orchestrator.

    Its cycle budget is :attr:`Settings.max_cycles` (``0`` means unbounded), so a
    session can be given a finite budget through ``FURROW_MAX_CYCLES`` or the
    ``--cycles`` flag on ``furrow start``.
    """
    return Orchestrator(goal=goal, on_event=on_event)


class EventStream:
    """Queue-backed bridge between the synchronous sink and the socket."""

    def __init__(self, websocket: WebSocketLike, *, maxsize: int = QUEUE_SIZE) -> None:
        self._websocket = websocket
        self._queue: asyncio.Queue[EventPayload | None] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def emit(self, event: str, payload: EventPayload) -> None:
        """Event sink half; runs synchronously inside the orchestrator loop.

        ``event`` is already part of ``payload``; it is accepted only to match
        the :data:`Orchestrator` sink signature.
        """
        self._offer(payload)

    def finish(self, *, reason: str, cycles: int = 0) -> None:
        """Queue the end-of-session marker and the sentinel that stops the pump."""
        self._offer(
            {
                "event": "session_end",
                "message": f"Session ended ({reason}) after {cycles} cycle(s).",
                "reason": reason,
                "cycles": cycles,
                "dropped": self.dropped,
            }
        )
        self._offer(None)

    async def pump(self) -> None:
        """Sender half; forwards queued payloads until the sentinel arrives."""
        while True:
            payload = await self._queue.get()
            try:
                if payload is None:
                    return
                await self._websocket.send_json(payload)
            except Exception:  # any socket/driver error ends the stream, nothing else
                logger.debug("furrow: websocket write failed, ending stream", exc_info=True)
                return
            finally:
                self._queue.task_done()

    def _offer(self, payload: EventPayload | None) -> None:
        """Never blocks; drops the oldest event rather than the newest."""
        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:
            self.dropped += 1
            with contextlib.suppress(asyncio.QueueEmpty, asyncio.QueueFull):
                self._queue.get_nowait()
                self._queue.put_nowait(payload)


async def _watch_client(websocket: WebSocketLike) -> str:
    """Wait for client control messages; return why the session should end."""
    while True:
        try:
            message = await websocket.receive_json()
        except WebSocketDisconnect:
            return "client_disconnected"
        except (ValueError, TypeError, RuntimeError):
            # Malformed frame or a socket that is already gone.
            return "client_disconnected"
        if not isinstance(message, dict):
            continue
        if str(message.get("action", "")).strip().lower() in STOP_ACTIONS:
            return "client_stopped"


async def _teardown(
    stream: EventStream,
    orchestrator: Orchestrator,
    runner: asyncio.Task[None],
    watcher: asyncio.Task[str],
    pump: asyncio.Task[None],
    *,
    reason: str,
) -> None:
    """Stop the agent, flush the last events, and leave no task behind."""
    if not runner.done():
        runner.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await runner
    if runner.done() and not runner.cancelled() and runner.exception() is not None:
        logger.warning("furrow: orchestrator run failed: %r", runner.exception())
        stream.emit(
            "error",
            {
                "event": "error",
                "message": f"The run ended with an error: {runner.exception()}",
                "reason": "run_error",
            },
        )

    cycles = int(getattr(orchestrator, "cycles", 0) or 0)
    stream.finish(reason=reason, cycles=cycles)
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(asyncio.shield(pump), timeout=DRAIN_TIMEOUT)
    if not pump.done():
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump
    watcher.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await watcher


async def run_session(
    websocket: WebSocketLike,
    goal: str,
    *,
    make_orchestrator: OrchestratorFactory | None = None,
) -> str:
    """Run one orchestrator session, streaming its events to ``websocket``.

    Returns the reason the session ended (``finished``, ``client_stopped`` or
    ``client_disconnected``).  It does not raise for socket or run failures; both
    are reported as events and as log records instead.
    """
    stream = EventStream(websocket)
    factory = make_orchestrator or _default_orchestrator
    orchestrator = factory(goal, stream.emit)
    pump = asyncio.create_task(stream.pump(), name=PUMP_TASK)
    runner = asyncio.create_task(orchestrator.run(), name=RUNNER_TASK)
    watcher = asyncio.create_task(_watch_client(websocket), name=WATCHER_TASK)

    reason = "finished"
    try:
        done, _pending = await asyncio.wait(
            {runner, watcher}, return_when=asyncio.FIRST_COMPLETED
        )
        if runner not in done:
            reason = await watcher
    except asyncio.CancelledError:
        await _teardown(
            stream, orchestrator, runner, watcher, pump, reason="server_cancelled"
        )
        raise
    except Exception:
        logger.exception("furrow: session supervision failed")
        reason = "session_error"
    await _teardown(stream, orchestrator, runner, watcher, pump, reason=reason)
    return reason


async def _send(websocket: WebSocketLike, payload: EventPayload) -> None:
    with contextlib.suppress(Exception):
        await websocket.send_json(payload)


async def _error(websocket: WebSocketLike, message: str) -> None:
    await _send(websocket, {"event": "error", "message": message, "reason": "bad_request"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        start = StartRequest.model_validate(await websocket.receive_json())
    except WebSocketDisconnect:
        return
    except ValidationError:
        await _error(websocket, 'Start message must be {"goal": "..."}.')
    except (ValueError, TypeError):
        await _error(websocket, "Start message must be valid JSON.")
    else:
        await run_session(websocket, start.goal)
    with contextlib.suppress(Exception):
        await websocket.close()


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Furrow</title>
<style>
  :root { color-scheme: dark; }
  body {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: #10131a; color: #d7dce5;
    margin: 0 auto; padding: 1.5rem; max-width: 62rem;
  }
  h1 { font-size: 1.1rem; letter-spacing: 0.35em; color: #7ee787; }
  p.hint { color: #8b949e; font-size: 0.8rem; margin: 0 0 1rem; }
  form { display: flex; gap: 0.5rem; margin-bottom: 0.6rem; }
  input {
    flex: 1; padding: 0.5rem; background: #171b24; color: inherit;
    border: 1px solid #2b3240; border-radius: 4px;
  }
  button {
    padding: 0.5rem 1rem; border: 0; border-radius: 4px; cursor: pointer;
    background: #238636; color: #ffffff; font: inherit;
  }
  button#stop { background: #6e2c28; }
  button[disabled] { opacity: 0.45; cursor: not-allowed; }
  #status { font-size: 0.8rem; color: #8b949e; margin-bottom: 0.5rem; }
  #out {
    background: #0b0e14; border: 1px solid #2b3240; border-radius: 4px;
    padding: 0.7rem; min-height: 18rem; max-height: 70vh; overflow: auto;
    font-size: 0.82rem; line-height: 1.45;
  }
  .line { display: block; white-space: pre-wrap; word-break: break-word; }
  .evt { color: #79c0ff; }
  .start, .cycle_start, .halt, .session_end { color: #d2a8ff; }
  .task_complete, .plan, .test_result { color: #7ee787; }
  .task_failed, .test_failure, .retry_planned, .error { color: #ff7b72; }
  .cycle_end, .parallel_limit, .closed, .raw { color: #8b949e; }
</style>
</head>
<body>
  <h1>FURROW</h1>
  <p class="hint">
    Runs until the goal is met, Stop is pressed, or FURROW_MAX_CYCLES is reached.
  </p>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit" id="start">Start</button>
    <button type="button" id="stop" disabled>Stop</button>
  </form>
  <div id="status">idle</div>
  <div id="out"></div>
<script>
  const form = document.getElementById('form');
  const out = document.getElementById('out');
  const statusEl = document.getElementById('status');
  const startBtn = document.getElementById('start');
  const stopBtn = document.getElementById('stop');
  let socket = null;

  function write(name, text) {
    const line = document.createElement('span');
    line.className = 'line';
    const tag = document.createElement('span');
    tag.className = 'evt';
    tag.textContent = '[' + name + '] ';
    line.appendChild(tag);
    line.appendChild(document.createTextNode(text == null ? '' : String(text)));
    out.appendChild(line);
    out.scrollTop = out.scrollHeight;
  }

  form.onsubmit = (ev) => {
    ev.preventDefault();
    if (socket) { return; }
    const goal = document.getElementById('goal').value.trim();
    if (!goal) { return; }
    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusEl.textContent = 'connecting...';
    socket = new WebSocket('ws://' + location.host + '/ws');
    socket.onopen = () => {
      statusEl.textContent = 'running...';
      socket.send(JSON.stringify({ goal: goal }));
    };
    socket.onmessage = (ev) => {
      let payload;
      try {
        payload = JSON.parse(ev.data);
      } catch (err) {
        write('raw', ev.data);
        return;
      }
      const name = payload.event || 'event';
      write(name, payload.message);
      if (name === 'cycle_start' && payload.cycle) {
        statusEl.textContent = 'running... cycle ' + payload.cycle;
      }
      if (name === 'task_start' && payload.task_id) {
        statusEl.textContent = 'running... ' + payload.task_id;
      }
      if (name === 'session_end' || name === 'halt') {
        statusEl.textContent = 'finishing...';
      }
    };
    socket.onerror = () => write('error', 'websocket error');
    socket.onclose = () => {
      socket = null;
      startBtn.disabled = false;
      stopBtn.disabled = true;
      statusEl.textContent = 'idle';
      write('closed', 'Session closed.');
    };
  };

  stopBtn.onclick = () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      statusEl.textContent = 'stopping...';
      socket.send(JSON.stringify({ action: 'stop' }));
    }
  };
</script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
