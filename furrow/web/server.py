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


class _WSWriter:
    """File-like sink that forwards Rich console output to a WebSocket client."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._loop = asyncio.get_running_loop()

    def write(self, s: str) -> int:
        if s:
            self._loop.create_task(self._send(s))
        return len(s)

    def flush(self) -> None:
        pass

    async def _send(self, s: str) -> None:
        try:
            await self._ws.send_text(s)
        except Exception:  # noqa: BLE001
            pass


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
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        ws_console = Console(
            file=_WSWriter(websocket),
            force_terminal=False,
            highlight=False,
            width=120,
        )
        orchestrator = Orchestrator(goal=goal, console=ws_console)
        await orchestrator.run()
        await websocket.send_text("\n[stream ended]\n")
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
