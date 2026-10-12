from __future__ import annotations

import asyncio
import sys
from typing import Optional
from io import StringIO

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console
from rich.text import Text

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketConsole(Console):
    """A Console that writes to a WebSocket instead of stdout."""

    def __init__(self, websocket: WebSocket) -> None:
        super().__init__(file=StringIO(), force_terminal=True, width=120)
        self.websocket = websocket
        self._buffer = ""

    def print(self, *args, **kwargs) -> None:
        # Capture the output
        super().print(*args, **kwargs)
        # Get the content and send it
        content = self.file.getvalue()
        if content != self._buffer:
            new_content = content[self._buffer :]
            self._buffer = content
            # Send asynchronously
            asyncio.create_task(self._send(new_content))

    async def _send(self, message: str) -> None:
        try:
            await self.websocket.send_text(message)
        except Exception:
            pass  # Connection closed


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 1rem; }
        #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; min-height: 400px; white-space: pre-wrap; }
        .green { color: #4ec9b0; }
        .yellow { color: #dcdcaa; }
        .red { color: #f44747; }
        .cyan { color: #9cdcfe; }
        .blue { color: #569cd6; }
        input { padding: 0.5rem; font-size: 1rem; width: 70%; }
        button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
    </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    let ws = null;

    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      out.textContent += '\\n> Starting: ' + goal + '\\n';
      
      if (ws) ws.close();
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data;
      ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
      ws.onopen = () => ws.send(JSON.stringify({goal: goal}));
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

        # Create a custom console that writes to the websocket
        ws_console = WebSocketConsole(websocket)

        # Monkey-patch the console in the orchestrator module
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
    except Exception as e:
        try:
            await websocket.send_text(f"\\n[Error: {e}]\\n")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)