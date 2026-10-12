from __future__ import annotations

import asyncio
from io import StringIO

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator
from furrow.logging import get_logger

app = FastAPI(title="Furrow")
logger = get_logger(__name__)


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class WebSocketConsole(Console):
    def __init__(self, websocket: WebSocket, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.websocket = websocket
        self.output_buffer = StringIO()

    def print(self, *args, **kwargs):
        # Capture output to buffer
        self.output_buffer.write(" ".join(str(arg) for arg in args) + "\n")
        # Also send to websocket
        message = " ".join(str(arg) for arg in args)
        asyncio.create_task(self._send_message(message))

    async def _send_message(self, message: str):
        try:
            await self.websocket.send_text(message)
        except (WebSocketDisconnect, OSError):
            logger.debug("websocket_send_failed", reason="connection_closed")


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
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
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

        # Create a custom console that sends output to websocket
        ws_console = WebSocketConsole(websocket, force_terminal=True, color_system="truecolor")

        # We need to patch the orchestrator's console
        import furrow.core.orchestrator as orchestrator_module

        original_console = orchestrator_module.console
        orchestrator_module.console = ws_console

        try:
            orchestrator = Orchestrator(goal=goal)
            await orchestrator.run()
        finally:
            orchestrator_module.console = original_console
    except WebSocketDisconnect:
        pass
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("websocket_handler_error", error=str(e))
        try:
            await websocket.send_text(f"Error: {e}")
        except (WebSocketDisconnect, OSError):
            logger.debug("websocket_send_failed", reason="connection_closed")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)