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

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


INDEX_HTML = """<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
  h1 { color: #2e7d32; }
  #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 6px; font-family: monospace; white-space: pre-wrap; height: 500px; overflow-y: auto; }
  #form { margin: 1rem 0; }
  #goal { width: 100%; padding: 0.5rem; font-size: 1rem; }
  button { padding: 0.5rem 1rem; background: #2e7d32; color: white; border: none; border-radius: 4px; cursor: pointer; }
  button:hover { background: #1b5e20; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal..." required />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = 'Starting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onopen = () => ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    async def send_log(message: str) -> None:
        """Callback that streams console output to the WebSocket client."""
        try:
            await websocket.send_text(message)
        except Exception:
            pass

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        settings = Settings()
        if data.get("model"):
            settings.model = data["model"]

        orchestrator = Orchestrator(
            goal=goal,
            settings=settings,
            log_callback=send_log,
        )
        await orchestrator.run()
        await websocket.send_text("\n[bold green]Furrow finished.[/bold green]")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"\n[red]Error: {e}[/red]")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
