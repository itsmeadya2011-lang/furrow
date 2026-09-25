from __future__ import annotations

import asyncio
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          out.textContent += JSON.stringify(data) + '\\n';
        } catch (e) {
          out.textContent += ev.data + '\\n';
        }
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onerror = (e) => out.textContent += '\\nError: ' + e.message + '\\n';
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

        if model:
            from furrow.config import settings
            settings.model = model
            settings.planner_model = model
            settings.worker_model = model
            settings.tester_model = model

        async def send_event(event: dict[str, Any]) -> None:
            await websocket.send_json(event)

        orchestrator = Orchestrator(goal=goal, event_callback=send_event)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "error": str(e)})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
