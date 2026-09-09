from __future__ import annotations

import asyncio
from typing import Optional

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
      ws.onopen    = () => ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      ws.onmessage = (ev) => { out.textContent += ev.data + '\\n'; out.scrollTop = out.scrollHeight; };
      ws.onclose   = () => out.textContent += '\\nClosed.\\n';
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()

    async def send(message: str) -> None:
        try:
            await websocket.send_text(message)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    def on_output(message: str) -> None:
        asyncio.run_coroutine_threadsafe(send(message), loop)

    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    goal = data.get("goal", "")
    try:
        await websocket.send_text("started")
        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run(on_output=on_output)
        await websocket.send_text("completed")
    except WebSocketDisconnect:
        try:
            await websocket.send_text("stopped")
        except WebSocketDisconnect:
            pass
    except Exception as e:
        try:
            await websocket.send_text(f"error: {e}")
        except WebSocketDisconnect:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
