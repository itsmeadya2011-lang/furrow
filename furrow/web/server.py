from __future__ import annotations

import asyncio
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from rich.console import Console

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketConsole:
    """A file-like object that sends text to a WebSocket."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    def write(self, text: str) -> int:
        if text.strip():
            try:
                asyncio.get_running_loop().create_task(self._websocket.send_text(text.rstrip()))
            except RuntimeError:
                pass
        return len(text)

    def flush(self) -> None:
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
    <input id="model" placeholder="Model (optional)" />
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
      const payload = {
        goal: document.getElementById('goal').value,
        model: document.getElementById('model').value || null
      };
      ws.send(JSON.stringify(payload));
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
        settings = Settings()
        if model:
            settings.model = model
        ws_console = Console(file=WebSocketConsole(websocket), force_terminal=True)
        orchestrator = Orchestrator(goal=goal, settings=settings, console=ws_console)
        await orchestrator.run()
        await websocket.send_text("\n[bold green]Furrow session complete.[/bold green]\n")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"\n[bold red]Error: {e}[/bold red]\n")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
