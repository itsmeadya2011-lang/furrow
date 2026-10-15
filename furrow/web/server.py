from __future__ import annotations

import asyncio
import logging
from io import StringIO

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")
logger = logging.getLogger(__name__)


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


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


class WebSocketStream(StringIO):
    """Redirects stdout to a WebSocket."""
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket

    def write(self, data: str) -> int:
        super().write(data)
        # Send complete lines
        content = self.getvalue()
        lines = content.split('\n')
        if len(lines) > 1:
            # Send all complete lines
            for line in lines[:-1]:
                asyncio.create_task(self.websocket.send_text(line))
            # Keep the last incomplete line in buffer
            self.seek(0)
            self.truncate(0)
            self.write(lines[-1])
        return len(data)

    def flush(self) -> None:
        content = self.getvalue()
        if content:
            asyncio.create_task(self.websocket.send_text(content))
            self.seek(0)
            self.truncate(0)

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal:
            await websocket.send_text("Error: No goal provided")
            return

        # Create a custom console that writes to websocket
        from rich.console import Console
        stream = WebSocketStream(websocket)
        console = Console(file=stream, force_terminal=False, width=120)

        # Patch the global console in orchestrator
        import furrow.core.orchestrator as orch_module
        original_console = orch_module.console
        orch_module.console = console

        try:
            orchestrator = Orchestrator(goal=goal)
            await orchestrator.run()
        finally:
            orch_module.console = original_console
            stream.flush()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception:  # noqa: BLE001
            logger.debug("Failed to send error to websocket: %s", e)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
