from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from furrow.config import get_settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")

# Queued to tell the sender task that no further events are coming. A sentinel
# (rather than a flag) keeps every send strictly ordered.
_SENTINEL: dict[str, Any] = {}


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class _Run:
    """Bookkeeping for one orchestrator run streamed over one socket.

    ``Orchestrator`` takes a *synchronous* ``on_event`` callback, so events are
    buffered on an unbounded queue and flushed by a dedicated sender task. The
    orchestrator therefore never blocks on the socket, and event order is
    preserved even though the browser is a slow consumer.
    """

    def __init__(self, run_id: str, websocket: WebSocket) -> None:
        self.id = run_id
        self.websocket = websocket
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.task: Optional[asyncio.Task[None]] = None
        self.stop_requested = False
        self.connected = True

    def publish(self, event: dict[str, Any]) -> None:
        """``on_event`` callback handed to the orchestrator."""
        try:
            self.events.put_nowait(dict(event))
        except Exception as exc:  # a UI hiccup must never abort the run
            logger.warning("dropping event for run %s: %s", self.id, exc)

    def emit(self, event: dict[str, Any]) -> None:
        """Queue a server generated event (errors, terminal marker)."""
        self.events.put_nowait(event)

    def close_stream(self) -> None:
        """Signal the sender task that no further events are coming."""
        self.events.put_nowait(_SENTINEL)

    def request_stop(self) -> None:
        """Cancel the run. Idempotent, so repeated stop requests are harmless."""
        self.stop_requested = True
        task = self.task
        if task is not None and not task.done():
            task.cancel()


# run_id -> run, so ``POST /stop`` can reach a run started from a socket.
_active_runs: dict[str, _Run] = {}


async def _send(websocket: WebSocket, event: dict[str, Any]) -> bool:
    """Send one event as JSON, returning ``False`` when the client is gone.

    Every send goes through here so a disconnecting browser can neither raise
    out of the sender task nor leave the run hanging.
    """
    try:
        await websocket.send_json(event)
    except Exception as exc:  # includes WebSocketDisconnect / RuntimeError
        logger.debug("send failed: %s", exc)
        return False
    return True


async def _flush(active: _Run) -> None:
    """Forward queued events to the client until the stream is closed."""
    while True:
        event = await active.events.get()
        if event is _SENTINEL:
            return
        if active.connected and not await _send(active.websocket, event):
            active.connected = False
            active.request_stop()


async def _listen(active: _Run) -> None:
    """Watch inbound frames until the client stops the run or disconnects."""
    while True:
        message = await active.websocket.receive()
        if message.get("type") == "websocket.disconnect":
            active.connected = False
            active.request_stop()
            return
        text = message.get("text")
        if not text:
            continue
        try:
            payload = json.loads(text)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "stop":
            active.request_stop()
            return


async def _stream(active: _Run, orchestrator: Orchestrator) -> None:
    """Run the orchestrator, streaming events until it finishes or is stopped.

    The orchestrator runs as a task so a stop request can cancel it without any
    cooperation from the orchestrator itself. ``CancelledError`` is turned back
    into a ``stopped`` event rather than being allowed to escape.
    """
    active.task = asyncio.create_task(orchestrator.run(), name=f"furrow-run-{active.id}")
    sender = asyncio.create_task(_flush(active), name=f"furrow-send-{active.id}")
    listener = asyncio.create_task(_listen(active), name=f"furrow-listen-{active.id}")
    try:
        try:
            await active.task
        except asyncio.CancelledError:
            if not active.stop_requested:
                raise  # cancelled from outside (shutdown): unwind untouched
            active.emit({"type": "stopped", "reason": "cancelled"})
        except Exception as exc:
            logger.exception("run %s failed", active.id)
            active.emit({"type": "error", "message": str(exc) or type(exc).__name__})
            active.emit({"type": "stopped", "reason": "error"})
        else:
            active.emit({"type": "stopped", "reason": "finished"})
        active.close_stream()
        await sender
    finally:
        active.request_stop()
        listener.cancel()
        sender.cancel()
        for leftover in (listener, sender, active.task):
            if leftover is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await leftover
        _active_runs.pop(active.id, None)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe, plus a count of runs currently streaming."""
    return {"status": "ok", "active_runs": len(_active_runs)}


@app.post("/stop")
async def stop_run(run_id: Optional[str] = None) -> dict[str, Any]:
    """Stop one run (``?run_id=...``) or, by default, every active run."""
    if run_id is None:
        targets = list(_active_runs.values())
    else:
        found = _active_runs.get(run_id)
        targets = [found] if found is not None else []
    for active in targets:
        active.request_stop()
    return {"requested": len(targets), "stopped": [active.id for active in targets]}


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Furrow</title>
<style>
  body { margin: 2rem; background: #10131a; color: #d7dae0;
         font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  h1 { font-size: 1rem; letter-spacing: .1em; text-transform: uppercase; color: #7fd1a0; }
  form { display: flex; gap: .5rem; margin-bottom: 1rem; }
  input, button { font: inherit; padding: .5rem .75rem; border-radius: .35rem;
                  border: 1px solid #2c323d; background: #171b23; color: inherit; }
  input { flex: 1; }
  button { cursor: pointer; }
  button.stop { border-color: #7a3b3b; color: #f0a3a3; }
  button[disabled] { opacity: .4; cursor: not-allowed; }
  #out { white-space: pre-wrap; overflow: auto; max-height: 60vh; min-height: 12rem;
         line-height: 1.5; padding: .75rem; border: 1px solid #2c323d;
         border-radius: .35rem; background: #0b0d12; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" autocomplete="off" required />
    <button type="submit" id="go">Start</button>
    <button type="button" id="stop" class="stop" disabled>Stop</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const goalInput = document.getElementById('goal');
    const goBtn = document.getElementById('go');
    const stopBtn = document.getElementById('stop');

    function line(text) {
      out.textContent += text + '\n';
      out.scrollTop = out.scrollHeight;
    }

    // One orchestrator event -> one readable line.
    function describe(ev) {
      if (!ev || typeof ev !== 'object') return String(ev);
      let head = ev.type || 'event';
      if (ev.cycle !== undefined && ev.cycle !== null) head += ' #' + ev.cycle;
      const bits = [];
      const count = typeof ev.tasks === 'number' ? ev.tasks
                  : (typeof ev.count === 'number' ? ev.count : ev.task_count);
      if (typeof count === 'number') bits.push(count + ' task(s)');
      if (ev.goal) bits.push('goal: ' + ev.goal);
      if (ev.id !== undefined) bits.push('id ' + ev.id);
      if (ev.status) bits.push(ev.status);
      if (typeof ev.passed === 'boolean') bits.push(ev.passed ? 'tests passed' : 'tests failed');
      if (ev.summary) bits.push(ev.summary);
      if (ev.reason) bits.push(ev.reason);
      if (ev.message) bits.push(ev.message);
      return bits.length ? head + ' - ' + bits.join(' - ') : head;
    }

    let ws = null;

    function setRunning(running) {
      goBtn.disabled = running;
      goalInput.disabled = running;
      stopBtn.disabled = !running;
    }

    form.onsubmit = (e) => {
      e.preventDefault();
      const goal = goalInput.value.trim();
      if (!goal) {
        line('Goal must not be empty.');
        return;
      }
      out.textContent = '';
      const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
      ws = new WebSocket(scheme + location.host + '/ws');
      setRunning(true);
      line('Starting: ' + goal);
      ws.onopen = () => ws.send(JSON.stringify({ goal: goal }));
      ws.onmessage = (msg) => {
        let ev;
        try {
          ev = JSON.parse(msg.data);
        } catch (err) {
          line('unparsed: ' + msg.data);
          return;
        }
        line(describe(ev));
        if (ev.type === 'stopped' || ev.type === 'error') setRunning(false);
      };
      ws.onclose = () => {
        setRunning(false);
        line('Closed.');
      };
      ws.onerror = () => {
        setRunning(false);
        line('Connection error.');
      };
    };

    stopBtn.onclick = () => {
      line('Stop requested.');
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'stop' }));
      }
      fetch('/stop', { method: 'POST' }).catch(() => {});
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Stream one orchestrator run to the browser.

    Client to server: ``{"goal": str, "model": str?}`` to start, then
    ``{"type": "stop"}`` at any time to halt the run. Server to client: one JSON
    object per orchestrator event, always ending with
    ``{"type": "stopped", "reason": ...}``.
    """
    try:
        await websocket.accept()
    except Exception as exc:
        logger.debug("accept failed: %s", exc)
        return

    try:
        request = StartRequest.model_validate(await websocket.receive_json())
    except WebSocketDisconnect:
        return
    except ValidationError as exc:
        await _send(websocket, {"type": "error", "message": f"Invalid request: {exc}"})
        return
    except ValueError as exc:  # body was not valid JSON
        await _send(websocket, {"type": "error", "message": f"Invalid request: {exc}"})
        return

    goal = request.goal.strip()
    if not goal:
        await _send(websocket, {"type": "error", "message": "Goal must not be empty."})
        return

    settings = get_settings()
    if request.model:
        settings = settings.model_copy(update={"model": request.model})

    active = _Run(uuid4().hex, websocket)
    try:
        orchestrator = Orchestrator(
            goal=goal,
            client=LLMClient(settings=settings),
            on_event=active.publish,
        )
    except Exception as exc:
        logger.exception("could not start run")
        await _send(websocket, {"type": "error", "message": str(exc) or type(exc).__name__})
        return

    _active_runs[active.id] = active
    try:
        await _stream(active, orchestrator)
    except WebSocketDisconnect:
        logger.debug("run %s: client disconnected", active.id)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
