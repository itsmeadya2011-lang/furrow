from __future__ import annotations

import asyncio
import traceback
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, settings
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
    <input id="model" placeholder="Model (optional)" />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const renderMessage = (msg) => {
      if (msg && typeof msg === 'object' && msg.type) {
        if (msg.type === 'error') {
          out.textContent += '[ERROR] ' + (msg.message || JSON.stringify(msg)) + '\\n';
        } else if (msg.type === 'status') {
          out.textContent += '[STATUS] ' + (msg.message || '') + '\\n';
        } else {
          out.textContent += '[' + msg.type.toUpperCase() + '] ' + JSON.stringify(msg) + '\\n';
        }
      } else {
        out.textContent += String(msg) + '\\n';
      }
    };
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      const model = document.getElementById('model').value;
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          renderMessage(JSON.parse(ev.data));
        } catch {
          renderMessage(ev.data);
        }
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onerror = () => out.textContent += '\\nWebSocket error.\\n';
      ws.onopen = () => {
        ws.send(JSON.stringify({goal, model: model || null}));
      };
    };
  </script>
</body>
</html>
""")


async def _send_json(websocket: WebSocket, type_: str, message: str = "", **extra: object) -> None:
    payload: dict[str, object] = {"type": type_, "message": message}
    payload.update(extra)
    try:
        await websocket.send_json(payload)
    except Exception:
        pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return
    except Exception as e:
        await _send_json(websocket, "error", f"Failed to read payload: {e}")
        await websocket.close()
        return

    goal = data.get("goal", "")
    model = data.get("model")

    if not goal:
        await _send_json(websocket, "error", "Missing required field: goal")
        await websocket.close()
        return

    await _send_json(websocket, "status", f"Starting orchestrator for goal: {goal}")

    run_settings = settings
    if model:
        run_settings = Settings(**settings.model_dump())
        run_settings.model = model
        run_settings.planner_model = model
        run_settings.worker_model = model
        run_settings.tester_model = model

    try:
        orchestrator = Orchestrator(goal=goal, settings=run_settings)
        await _send_json(websocket, "status", "Orchestrator running...")
        await orchestrator.run()
        await _send_json(websocket, "status", "Orchestrator finished")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        tb = traceback.format_exc()
        await _send_json(websocket, "error", f"{type(e).__name__}: {e}", traceback=tb)
        try:
            await websocket.close()
        except Exception:
            pass
    else:
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
