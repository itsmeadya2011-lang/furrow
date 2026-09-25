from __future__ import annotations

import asyncio
import contextlib

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator
from furrow.reporter import QueueReporter, drain_queue

app = FastAPI(title="Furrow")


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
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return
    goal = data.get("goal", "")
    if not goal:
        await websocket.send_text("Error: 'goal' is required")
        await websocket.close()
        return

    queue: asyncio.Queue[str] = asyncio.Queue()
    reporter = QueueReporter(queue)
    orchestrator = Orchestrator(goal=goal, reporter=reporter)
    drain_task = asyncio.create_task(drain_queue(queue, websocket.send_text))

    try:
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    finally:
        # Flush anything still queued before tearing down the drain.
        with contextlib.suppress(Exception):
            while not queue.empty():
                await websocket.send_text(queue.get_nowait())
        drain_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drain_task
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
