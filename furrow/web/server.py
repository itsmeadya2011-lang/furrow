from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# The orchestrator currently executing (if any). Tracked at module level so the
# /stop endpoints can signal it.
_ACTIVE: Orchestrator | None = None

EventSink = Callable[[str], Awaitable[None]]


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


def _build_orchestrator(goal: str, on_event: EventSink) -> Orchestrator:
    """Create an Orchestrator, tolerating both ``on_event``-aware and legacy signatures.

    ``Orchestrator`` accepts sync and async sinks; the try/except keeps the web layer
    working against an orchestrator that predates the parameter entirely.
    """
    try:
        return Orchestrator(goal=goal, on_event=on_event)
    except TypeError:
        return Orchestrator(goal=goal)


def _make_sink(send: Callable[[str], Awaitable[None]], errors: list[str]) -> EventSink:
    """Build an ``on_event`` callback that forwards each line, surviving disconnects."""

    async def emit(line: Any) -> None:
        try:
            await send(str(line))
        except Exception as exc:  # noqa: BLE001 - client may vanish mid-run
            errors.append(str(exc))

    return emit


async def _send_text(websocket: WebSocket, line: str) -> None:
    """Send a line, swallowing the inevitable failure once the client is gone."""
    try:
        await websocket.send_text(line)
    except (WebSocketDisconnect, RuntimeError, OSError):
        pass


async def _stop_active() -> bool:
    """Signal the running orchestrator to halt. Returns True if one was signalled."""
    orchestrator = _ACTIVE
    if orchestrator is None:
        return False
    stop = getattr(orchestrator, "stop", None)
    if stop is None:
        return False
    try:
        result = stop()
        if hasattr(result, "__await__"):
            await result
    except Exception:  # noqa: BLE001 - stopping must never raise into the route
        return False
    return True


async def _execute(goal: str, emit: EventSink | None) -> str:
    """Run the orchestrator to completion, streaming events into ``emit``.

    Returns an empty string on success or a human readable error message.
    """
    global _ACTIVE  # single shared run slot by design
    sink: EventSink = emit if emit is not None else _make_sink(_noop_send, [])
    orchestrator = _build_orchestrator(goal, sink)
    previous, _ACTIVE = _ACTIVE, orchestrator
    try:
        if emit is None:
            await sink("note: orchestrator has no on_event support; streaming disabled")
        await orchestrator.run()
        return ""
    except Exception as exc:  # noqa: BLE001 - surfaced to the client instead
        return f"{type(exc).__name__}: {exc}"
    finally:
        if _ACTIVE is orchestrator:
            _ACTIVE = previous


async def _noop_send(_line: str) -> None:
    return None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8" /><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit" id="start">Start</button>
    <button type="button" id="stop">Stop</button>
  </form>
  <pre id="log"></pre>
  <script>
    const form = document.getElementById('form');
    const stopBtn = document.getElementById('stop');
    const log = document.getElementById('log');
    let ws = null;

    function append(line) {
      log.textContent += line + '\\n';
      log.scrollTop = log.scrollHeight;
    }

    stopBtn.onclick = async () => {
      append('Stopping...');
      try {
        const res = await fetch('/stop', {method: 'POST'});
        append((await res.text()).trim() || 'stop requested');
      } catch (err) {
        append('stop failed: ' + err);
      }
    };

    form.onsubmit = (e) => {
      e.preventDefault();
      log.textContent = '';
      append('Starting...');
      const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(scheme + '://' + location.host + '/ws');
      ws.onmessage = (ev) => append(ev.data);
      ws.onclose = () => append('Closed.');
      ws.onerror = () => append('Connection error.');
      ws.onopen = () => ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    send_errors: list[str] = []
    try:
        payload = StartRequest.model_validate(await websocket.receive_json())
        emit = _make_sink(lambda line: _send_text(websocket, line), send_errors)
        await _send_text(websocket, f"running: {payload.goal}")
        error = await _execute(payload.goal, emit)
        if error:
            await _send_text(websocket, f"error: {error}")
        else:
            await _send_text(websocket, "done")
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - always answer the client
        await _send_text(websocket, f"error: {type(exc).__name__}: {exc}")
    finally:
        if send_errors:
            print(f"websocket send failed for client: {send_errors[-1]}")
        try:
            await websocket.close()
        except (RuntimeError, WebSocketDisconnect, OSError):
            pass


@app.post("/stop")
async def stop_post() -> dict[str, Any]:
    stopped = await _stop_active()
    return {"stopped": stopped, "active": _ACTIVE is not None}


@app.get("/stop")
async def stop_get() -> dict[str, Any]:
    stopped = await _stop_active()
    return {"stopped": stopped, "active": _ACTIVE is not None}


@app.post("/api/start")
async def api_start(payload: StartRequest, background: BackgroundTasks) -> dict[str, Any]:
    if _ACTIVE is not None:
        return {"started": False, "reason": "a run is already in progress"}
    background.add_task(_execute, payload.goal, None)
    return {"started": True, "goal": payload.goal}


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)