from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketWriter:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket

    def write(self, text: str) -> None:
        asyncio.create_task(self.websocket.send_text(text))

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
    <button type="submit">Start</button>
  </form>
  <pre id="out" style="background:#111;color:#eee;padding:1em;border-radius:4px;"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        out.textContent += ev.data + '\\n';
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onerror = (ev) => out.textContent += '\\nError: ' + ev.type + '\\n';
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
        if not goal:
            await websocket.send_text("[red]Error: goal is required[/red]")
            return

        settings = Settings()
        if model:
            settings.model = model
        writer = WebSocketWriter(websocket)
        console = Console(file=writer, force_terminal=True, width=120)
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal=goal, console=console, client=client)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"[red]Error: {e}[/red]")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
