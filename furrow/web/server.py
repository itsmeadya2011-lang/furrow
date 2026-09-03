from __future__ import annotations

import asyncio
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
    <button type="button" id="stop" disabled>Stop</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const stopBtn = document.getElementById('stop');
    let ws = null;
    form.onsubmit = async (e) => {
      e.preventDefault();
      if (ws) return;
      out.textContent = '';
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => {
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
        stopBtn.disabled = false;
      };
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => {
        out.textContent += '\\nClosed.\\n';
        stopBtn.disabled = true;
        ws = null;
      };
    };
    stopBtn.onclick = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send('stop');
      }
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    cancel_event = asyncio.Event()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)
        run_task = asyncio.create_task(orchestrator.run())
        try:
            while not run_task.done() and not cancel_event.is_set():
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                    if message == "stop":
                        cancel_event.set()
                except asyncio.TimeoutError:
                    continue
                except WebSocketDisconnect:
                    cancel_event.set()
                    break
        finally:
            cancel_event.set()
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
