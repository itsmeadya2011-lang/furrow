from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class _WebSocketFile:
    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    def write(self, text: str) -> int:
        stripped = text.rstrip()
        if stripped:
            asyncio.create_task(self._websocket.send_text(stripped))
        return len(text)

    def flush(self) -> None:
        pass


class WebSocketConsole(Console):
    def __init__(self, websocket: WebSocket) -> None:
        super().__init__(file=_WebSocketFile(websocket), force_terminal=False, color_system=None)


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
      ws.onmessage = (ev) => { out.textContent += ev.data + '\\n'; out.scrollTop = out.scrollHeight; };
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
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        console = WebSocketConsole(websocket)
        orchestrator = Orchestrator(goal=goal, console=console)
        await orchestrator.run()
        await websocket.send_text("[DONE]")
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
