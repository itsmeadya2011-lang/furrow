from __future__ import annotations

import asyncio
import contextlib
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
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
    task: asyncio.Task | None = None
    try:
        await websocket.send_text("Starting...")
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        async def send_output(text: str) -> None:
            await websocket.send_text(text)

        orchestrator = Orchestrator(goal=goal, output_callback=send_output)
        task = asyncio.create_task(orchestrator.run())
        await task
    except WebSocketDisconnect:
        pass
    finally:
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
