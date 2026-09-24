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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        async def send_output(message: str) -> None:
            # Strip rich markup for web display
            clean = message.replace("[bold green]", "").replace("[/bold green]", "") \
                          .replace("[bold cyan]", "").replace("[/bold cyan]", "") \
                          .replace("[bold yellow]", "").replace("[/bold yellow]", "") \
                          .replace("[green]", "").replace("[/green]", "") \
                          .replace("[red]", "").replace("[/red]", "") \
                          .replace("[yellow]", "").replace("[/yellow]", "") \
                          .replace("[blue]", "").replace("[/blue]", "") \
                          .replace("═══", "===")
            await websocket.send_text(clean)
        
        orchestrator = Orchestrator(goal=goal, output_callback=send_output)
        await orchestrator.run()
        await websocket.send_text("\n[Done]")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"\n[Error] {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
