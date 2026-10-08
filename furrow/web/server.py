from __future__ import annotations

import asyncio
import logging

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
    h1 { color: #2c3e50; }
    form { margin-bottom: 1rem; }
    input { padding: 0.5rem; width: 70%; }
    button { padding: 0.5rem 1rem; background: #2c3e50; color: white; border: none; cursor: pointer; }
    button:disabled { background: #95a5a6; cursor: not-allowed; }
    #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; height: 60vh; overflow-y: auto; white-space: pre-wrap; }
    .cycle { color: #569cd6; font-weight: bold; }
    .plan { color: #4ec9b0; }
    .task { color: #dcdcaa; }
    .success { color: #4ec9b0; }
    .error { color: #f44747; }
    .warning { color: #dcdcaa; }
    .info { color: #9cdcfe; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit" id="btn">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const btn = document.getElementById('btn');
    let ws = null;

    function connect() {
      out.textContent += '\\n[Connecting...]\\n';
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => {
        out.textContent += '[Connected]\\n';
        btn.disabled = true;
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
      ws.onmessage = (ev) => {
        const msg = ev.data;
        out.textContent += msg + '\\n';
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => {
        out.textContent += '\\n[Disconnected]\\n';
        btn.disabled = false;
      };
      ws.onerror = (err) => {
        out.textContent += '\\n[Error: ' + err + ']\\n';
      };
    }

    form.onsubmit = (e) => {
      e.preventDefault();
      if (ws) ws.close();
      connect();
    };
  </script>
</body>
</html>
""")


class WebSocketConsole(Console):
    """Console that streams output to a WebSocket."""

    def __init__(self, websocket: WebSocket) -> None:
        super().__init__()
        self.websocket = websocket

    def print(self, *objects: object, **kwargs: object) -> None:  # type: ignore[override]
        """Send message to WebSocket."""
        message = " ".join(str(obj) for obj in objects)
        if message:
            asyncio.create_task(self._send(message))

    async def _send(self, message: str) -> None:
        try:
            await self.websocket.send_text(message)
        except (WebSocketDisconnect, RuntimeError):
            logger.debug("WebSocket connection closed while sending")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal:
            await websocket.send_text("[Error: No goal provided]")
            return

        # Create WebSocket-aware console
        ws_console = WebSocketConsole(websocket)

        # Monkey-patch the orchestrator's console
        import furrow.core.orchestrator as orch_module
        original_console = orch_module.console
        orch_module.console = ws_console

        try:
            orchestrator = Orchestrator(goal=goal)
            await orchestrator.run()
            await websocket.send_text("\n[Done]")
        finally:
            orch_module.console = original_console

    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001 - top-level handler reports any error to client
        try:
            await websocket.send_text(f"[Error: {e}]")
        except (WebSocketDisconnect, RuntimeError):
            logger.debug("Failed to send error message to WebSocket")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)