from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core import orchestrator as orch_module
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketOutput:
    """File-like object that forwards Rich console output to a WebSocket."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self._buffer = ""
        self._closed = False

    def write(self, text: str) -> int:
        if self._closed:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                asyncio.create_task(self._send(line))
        return len(text)

    def flush(self) -> None:
        pass

    async def _send(self, text: str) -> None:
        try:
            await self.websocket.send_text(text)
        except Exception:
            self._closed = True


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
    original_console = orch_module.console
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        ws_output = WebSocketOutput(websocket)
        ws_console = Console(
            file=ws_output,
            force_terminal=False,
            color_system=None,
            width=120,
        )

        # Redirect orchestrator output to the WebSocket
        orch_module.console = ws_console

        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    finally:
        orch_module.console = original_console


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
