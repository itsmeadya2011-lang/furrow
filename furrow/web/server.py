from __future__ import annotations

import asyncio
from io import StringIO
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
import furrow.core.orchestrator as orch_module

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
    await websocket.send_text("[FURROW_START]\n")
    try:
        data = await websocket.receive_json()
        try:
            request = StartRequest(**data)
        except ValidationError as e:
            await websocket.send_text(f'{{"error": "{str(e)}"}}')
            return
        goal = request.goal
        if not goal or not goal.strip():
            await websocket.send_text("[FURROW_ERROR] goal is required\n")
            return
        orchestrator = Orchestrator(goal=goal)
        recording_console = Console(file=StringIO(), record=True, force_terminal=False)
        original_console = orch_module.console
        orch_module.console = recording_console
        try:
            try:
                await orchestrator.run()
            except Exception as exc:
                await websocket.send_text(f"[FURROW_ERROR] {exc}\n")
            output = recording_console.export_text()
            await websocket.send_text(output)
        finally:
            orch_module.console = original_console
        await websocket.send_text("[FURROW_DONE]\n")
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
