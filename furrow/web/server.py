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
    provider: Optional[str] = None
    max_cycles: Optional[int] = None
    parallel: Optional[int] = None


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
        
        # Apply any settings from the request
        from furrow.config import settings
        if data.get("model"):
            settings.model = data["model"]
        if data.get("provider"):
            settings.provider = data["provider"]
        if data.get("max_cycles") is not None:
            settings.max_cycles = data["max_cycles"]
        if data.get("parallel") is not None:
            settings.max_parallel_tasks = data["parallel"]
        
        # Create custom orchestrator that streams to websocket
        orchestrator = Orchestrator(goal=goal)
        
        # Monkey-patch console output to send to websocket
        original_print = orchestrator.console.print if hasattr(orchestrator, 'console') else None
        
        async def send_output(message: str):
            await websocket.send_text(message)
        
        # Run orchestrator with streaming
        await orchestrator.run()
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"Error: {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
