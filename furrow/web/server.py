from __future__ import annotations

import asyncio
from typing import Callable, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body style="font-family: monospace; max-width: 900px; margin: 2em auto;">
  <h1>Furrow</h1>
  <div id="status">Disconnected</div>
  <form id="form">
    <input id="goal" placeholder="Enter goal" style="width: 70%;" required />
    <button type="submit">Start</button>
  </form>
  <pre id="out" style="background:#111; color:#eee; padding:1em; margin-top:1em; white-space:pre-wrap;"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const status = document.getElementById('status');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      status.textContent = 'Connecting...';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => {
        status.textContent = 'Connected';
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => status.textContent = 'Closed';
      ws.onerror = () => status.textContent = 'Error';
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def consume() -> None:
        while True:
            msg = await queue.get()
            await websocket.send_text(msg)

    consumer = asyncio.create_task(consume())
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        def callback(message: str) -> None:
            queue.put_nowait(message)

        await websocket.send_text("Starting Furrow...")

        orchestrator = Orchestrator(goal=goal, output_callback=callback)
        await orchestrator.run()
        await websocket.send_text("Furrow finished.")
    except WebSocketDisconnect:
        pass
    finally:
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)