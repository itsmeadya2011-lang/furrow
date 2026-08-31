from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")
logger = logging.getLogger(__name__)


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
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
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def on_log(message: str) -> None:
        await queue.put(message)

    forward_task: asyncio.Task[None] | None = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        forward_task = asyncio.create_task(_forward_logs(websocket, queue))
        orchestrator = Orchestrator(goal=goal, on_log=on_log)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    finally:
        if forward_task:
            forward_task.cancel()
            try:
                await forward_task
            except asyncio.CancelledError:
                pass
        try:
            await websocket.close()
        except Exception:
            pass


async def _forward_logs(websocket: WebSocket, queue: asyncio.Queue[str]) -> None:
    while True:
        message = await queue.get()
        try:
            await websocket.send_text(message)
        except Exception:
            break


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=host, port=port)
