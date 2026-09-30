from __future__ import annotations

import asyncio
import contextlib

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Track active WebSocket sessions for status reporting and cleanup.
active_sessions: set[WebSocket] = set()


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <button id="stop" disabled>Stop</button>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const stopBtn = document.getElementById('stop');
    let ws;
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => {
        out.textContent += '\\nClosed.\\n';
        stopBtn.disabled = true;
      };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      stopBtn.disabled = false;
    };
    stopBtn.onclick = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send('stop');
        stopBtn.disabled = true;
      }
    };
  </script>
</body>
</html>
"""
    )


@app.get("/status")
async def status() -> dict[str, object]:
    return {"status": "ok", "active_sessions": len(active_sessions)}


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    active_sessions.add(websocket)
    orch_task: asyncio.Task | None = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        # Sync callback: the orchestrator's _emit() invokes this synchronously,
        # so we schedule the async WebSocket send onto the running event loop.
        def progress_callback(status: str) -> None:
            # Fire-and-forget: send scheduled on the running event loop.
            # Stored to suppress RUF006 (task reference must not be discarded).
            _ = asyncio.create_task(websocket.send_text(status))  # noqa: RUF006

        orchestrator = Orchestrator(goal=goal, progress_callback=progress_callback)
        orch_task = asyncio.create_task(orchestrator.run())

        # Concurrently listen for client messages (e.g. "stop").
        while orch_task is not None and not orch_task.done():
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if message.strip().lower() == "stop":
                orch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await orch_task
                break

        # Wait for orch_task to complete (or be cancelled).
        if orch_task is not None and not orch_task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await orch_task

        await websocket.send_text("DONE")
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        if orch_task is not None and not orch_task.done():
            orch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await orch_task
        active_sessions.discard(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
