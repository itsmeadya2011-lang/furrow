from __future__ import annotations

import asyncio
import json
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
    const stopBtn = document.getElementById('stop');
    const out = document.getElementById('out');
    let ws = null;
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => { out.textContent += '\\nClosed.\\n'; stopBtn.disabled = true; };
      ws.onerror = (ev) => { out.textContent += '\\nError connecting.\\n'; stopBtn.disabled = true; };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      stopBtn.disabled = false;
    };
    stopBtn.onclick = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({stop: true}));
      }
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    orchestrator: Orchestrator | None = None
    run_task: asyncio.Task | None = None

    async def send_output(text: str) -> None:
        try:
            await websocket.send_text(text)
        except Exception:
            pass

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal, on_output=send_output)
        run_task = asyncio.create_task(orchestrator.run())

        while True:
            message = await websocket.receive_json()
            if message.get("stop") and orchestrator is not None:
                orchestrator.stop()
                if run_task is not None:
                    await run_task
                await websocket.send_text("\nStopped by user.\n")
                break
    except WebSocketDisconnect:
        if orchestrator is not None:
            orchestrator.stop()
        if run_task is not None:
            try:
                await run_task
            except asyncio.CancelledError:
                pass
    except Exception:
        if orchestrator is not None:
            orchestrator.stop()
        if run_task is not None:
            try:
                await run_task
            except asyncio.CancelledError:
                pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
