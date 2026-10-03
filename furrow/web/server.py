from __future__ import annotations

import asyncio
import contextlib
import json

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from furrow.config import RunResult, settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

#: A single send must never wedge the run loop because a browser stopped reading.
SEND_TIMEOUT_SECONDS = 5.0
#: Grace period for the orchestrator's own shutdown before its task is cancelled outright.
SHUTDOWN_TIMEOUT_SECONDS = 10.0


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Furrow</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; max-width: 60rem; }
    #form { display: flex; gap: 0.5rem; }
    #goal { flex: 1; padding: 0.5rem; }
    #status { color: #b58900; margin: 0.5rem 0; }
    pre {
      background: #11151c; color: #d7dae0; padding: 1rem; border-radius: 6px;
      max-height: 70vh; overflow-y: auto; white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button id="start" type="submit">Start</button>
  </form>
  <p id="status" hidden>Running...</p>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const statusEl = document.getElementById('status');
    const startBtn = document.getElementById('start');

    function append(line) {
      out.textContent += line + '\\n';
      out.scrollTop = out.scrollHeight;
    }

    form.onsubmit = (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value.trim();
      if (!goal) {
        return;
      }
      out.textContent = '';
      statusEl.hidden = false;
      startBtn.disabled = true;
      append('Starting: ' + goal);
      const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
      const ws = new WebSocket(scheme + location.host + '/ws');
      ws.onopen = () => ws.send(JSON.stringify({goal: goal}));
      ws.onmessage = (ev) => append(ev.data);
      ws.onerror = () => append('[error] websocket connection failed');
      ws.onclose = () => {
        statusEl.hidden = true;
        startBtn.disabled = false;
        append('Disconnected.');
      };
    };
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


async def _safe_send(websocket: WebSocket, message: str) -> bool:
    """Send one text frame, tolerating a socket that has already gone away."""
    try:
        await asyncio.wait_for(websocket.send_text(message), SEND_TIMEOUT_SECONDS)
    except (WebSocketDisconnect, RuntimeError, OSError, asyncio.TimeoutError):
        return False
    return True


async def _safe_close(websocket: WebSocket) -> None:
    with contextlib.suppress(Exception):
        await websocket.close()


async def _receive_goal(websocket: WebSocket) -> str | None:
    """Return the first valid goal on the socket, or ``None`` if the client left.

    Unusable frames get a one-line explanation and the socket stays open for a retry,
    so a single malformed message cannot take the server down with it.
    """
    while True:
        try:
            message = await websocket.receive()
        except (WebSocketDisconnect, RuntimeError):
            return None
        if message.get("type") == "websocket.disconnect":
            return None

        text = message.get("text")
        if text is None:
            await _safe_send(websocket, "Error: send a text frame containing JSON.")
            continue

        try:
            payload = json.loads(text)
        except ValueError:
            await _safe_send(websocket, "Error: payload is not valid JSON.")
            continue

        if not isinstance(payload, dict):
            await _safe_send(websocket, "Error: payload must be a JSON object.")
            continue

        goal = payload.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            await _safe_send(websocket, "Error: 'goal' is required and must be a non-empty string.")
            continue

        return goal.strip()


async def _watch_disconnect(websocket: WebSocket, disconnected: asyncio.Event) -> None:
    """Block until the browser hangs up, then flag it so the run can be torn down."""
    while not disconnected.is_set():
        try:
            message = await websocket.receive()
        except (WebSocketDisconnect, RuntimeError):
            disconnected.set()
            return
        if message.get("type") == "websocket.disconnect":
            disconnected.set()
            return


async def _stop_run(orchestrator: Orchestrator, run_task: asyncio.Task[RunResult]) -> None:
    """Ask the orchestrator to stop, then bound the wait so no task is ever leaked."""
    with contextlib.suppress(Exception):
        await asyncio.wait_for(orchestrator.stop(), SHUTDOWN_TIMEOUT_SECONDS)
    if not run_task.done():
        run_task.cancel()
    try:
        await asyncio.wait_for(run_task, SHUTDOWN_TIMEOUT_SECONDS)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    except Exception:
        pass


def _summary(result: RunResult) -> str:
    """Closing line: why the run stopped and how many cycles it took."""
    cycles = "cycle" if result.cycles == 1 else "cycles"
    return f"--- stopped: {result.stopped_reason} after {result.cycles} {cycles} ---"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    goal = await _receive_goal(websocket)
    if goal is None:
        await _safe_close(websocket)
        return

    disconnected = asyncio.Event()

    async def send_progress(message: str) -> None:
        """Forward one progress event to the browser.

        The orchestrator awaits this per event. Sends are best effort: a client that
        navigated away must never be able to take the run down.
        """
        if disconnected.is_set():
            return
        if not await _safe_send(websocket, message):
            disconnected.set()

    orchestrator = Orchestrator(goal=goal, on_progress=send_progress)
    # run() is an unbounded loop, so it must not be awaited inline: the handler stays
    # responsive and progress streams out while the run is in flight.
    run_task = asyncio.create_task(orchestrator.run(), name="furrow-run")
    disconnect_watch = asyncio.create_task(
        _watch_disconnect(websocket, disconnected), name="furrow-disconnect-watch"
    )

    try:
        await asyncio.wait({run_task, disconnect_watch}, return_when=asyncio.FIRST_COMPLETED)
        if disconnected.is_set():
            return
        if run_task.done() and not run_task.cancelled():
            error = run_task.exception()
            if error is None:
                await send_progress(_summary(run_task.result()))
            else:
                await send_progress(f"--- run failed: {error} ---")
    finally:
        disconnect_watch.cancel()
        await _stop_run(orchestrator, run_task)
        await _safe_close(websocket)


def run(host: str | None = None, port: int | None = None) -> None:
    """Serve the web UI, defaulting to the configured loopback host and port."""
    resolved_host = settings.web_host if host is None else host
    resolved_port = settings.web_port if port is None else port
    uvicorn.run(app, host=resolved_host, port=resolved_port)
