from __future__ import annotations

import asyncio
from io import StringIO
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketConsole(Console):
    """A Rich Console that sends output to a WebSocket."""

    def __init__(self, websocket: WebSocket, **kwargs):
        super().__init__(**kwargs)
        self.websocket = websocket

    def _send(self, text: str) -> None:
        try:
            asyncio.get_event_loop().create_task(self.websocket.send_text(text))
        except Exception:
            pass

    def print(self, *args, **kwargs):
        with StringIO() as buf:
            tmp = Console(file=buf, force_terminal=False, no_color=True)
            tmp.print(*args, **kwargs)
            self._send(buf.getvalue())

    def log(self, *args, **kwargs):
        with StringIO() as buf:
            tmp = Console(file=buf, force_terminal=False, no_color=True)
            tmp.log(*args, **kwargs)
            self._send(buf.getvalue())


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
      ws.onopen = () => {
        out.textContent += 'Connected. Starting Furrow...\\n';
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
      ws.onmessage = (ev) => out.textContent += ev.data;
      ws.onclose = () => out.textContent += '\\nDone.\\n';
      ws.onerror = (err) => out.textContent += '\\nError: ' + err + '\\n';
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
        model = data.get("model")

        if model:
            settings = Settings()
            settings.model = model

        await websocket.send_text(f"Goal: {goal}\nStarting Furrow...\n\n")

        console = WebSocketConsole(websocket)
        orchestrator = Orchestrator(goal=goal, console=console)
        await orchestrator.run()

        await websocket.send_text("\n[Furrow] Completed.\n")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"\n[ERROR] {str(e)}\n")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)