from __future__ import annotations

import asyncio
import io
import json

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator
from furrow.logging import get_logger

log = get_logger(__name__)

app = FastAPI(title="Furrow")


class WebSocketConsole(io.StringIO):
    """A StringIO-like console that forwards writes to a WebSocket."""

    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket

    def write(self, data: str) -> int:
        # Write to buffer (parent StringIO)
        result = super().write(data)
        # Send data to websocket immediately
        if data:
            # Schedule the send on the event loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._send(data))
            except RuntimeError:
                # No running loop
                pass
        return result

    async def _send(self, data: str) -> None:
        try:
            await self.websocket.send_text(json.dumps({"type": "output", "data": data}))
        except (WebSocketDisconnect, RuntimeError) as e:
            log.debug("WebSocket send failed", error=str(e))

    def flush(self) -> None:
        pass


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
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
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'output') {
          out.textContent += msg.data;
        } else if (msg.type === 'started') {
          out.textContent += '\\n[Started]\\n';
        } else if (msg.type === 'completed') {
          out.textContent += '\\n[Completed]\\n';
        } else if (msg.type === 'error') {
          out.textContent += '\\n[Error] ' + msg.data + '\\n';
        }
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onopen = () => ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        # Create WebSocket console and send started message
        ws_console = WebSocketConsole(websocket)
        console = Console(file=ws_console, force_terminal=True, width=120)

        await websocket.send_text(json.dumps({"type": "started", "data": ""}))

        orchestrator = Orchestrator(goal=goal, console=console)
        await orchestrator.run()

        await websocket.send_text(json.dumps({"type": "completed", "data": ""}))
    except WebSocketDisconnect:
        pass
    except (RuntimeError, ValueError) as e:
        log.error("WebSocket error", error=str(e))
        try:
            await websocket.send_text(json.dumps({"type": "error", "data": str(e)}))
        except (WebSocketDisconnect, RuntimeError) as e2:
            log.debug("Failed to send error to websocket", error=str(e2))


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
