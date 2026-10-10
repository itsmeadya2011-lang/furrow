from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class _Streamer:
    """Minimal console-like object that forwards output over the websocket."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    def print(self, *args: Any, **kwargs: Any) -> None:
        text = " ".join(str(a) for a in args)
        asyncio.create_task(self._ws.send_text(text + "\n"))

    def log(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.print(message, *args, **kwargs)


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
      ws.onerror = () => out.textContent += '\\nError connecting.\\n';
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
        model = data.get("model")
        orchestrator = Orchestrator(
            goal=goal,
            max_cycles=data.get("max_cycles", 0),
            output=_Streamer(websocket),
        )
        if model:
            from furrow.config import settings
            settings.model = model
        await orchestrator.run()
        await websocket.send_text("\n[bold green]Furrow finished.[/bold green]\n")
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        with contextlib.suppress(Exception):
            await websocket.send_text(f"\n[red]Error: {e}[/red]")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
