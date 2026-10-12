from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Tracks active websocket sessions to prevent concurrent orchestrators on one connection.
_active_connections: set[WebSocket] = set()


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None
    planner_model: Optional[str] = None
    worker_model: Optional[str] = None
    tester_model: Optional[str] = None
    max_cycles: Optional[int] = None
    max_parallel: Optional[int] = None


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
    if websocket in _active_connections:
        await websocket.accept()
        await websocket.send_text("Furrow is already running on this connection.")
        await websocket.close()
        return

    _active_connections.add(websocket)
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal:
            await websocket.send_text("Error: goal is required.")
            return

        # Apply model/limit overrides to the global settings (matching the CLI).
        model = data.get("model")
        planner_model = data.get("planner_model")
        worker_model = data.get("worker_model")
        tester_model = data.get("tester_model")
        max_cycles = data.get("max_cycles")
        max_parallel = data.get("max_parallel")

        if model:
            settings.model = model
            settings.planner_model = model
            settings.worker_model = model
            settings.tester_model = model
        if planner_model:
            settings.planner_model = planner_model
        if worker_model:
            settings.worker_model = worker_model
        if tester_model:
            settings.tester_model = tester_model
        if max_cycles is not None:
            settings.max_cycles = max_cycles
        if max_parallel is not None:
            settings.max_parallel_tasks = max_parallel

        async def on_event(msg: str) -> None:
            await websocket.send_text(msg)

        orchestrator = Orchestrator(goal=goal, on_event=on_event)
        await orchestrator.run()
        await websocket.send_text("Furrow finished.")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception:
            pass
    finally:
        _active_connections.discard(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)