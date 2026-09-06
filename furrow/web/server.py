from __future__ import annotations

import asyncio
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
    """Serve the Furrow web client HTML page."""
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
    function append(type, message) {
      const line = document.createElement('div');
      if (type === 'error') {
        line.style.color = '#f88';
      } else if (type === 'status') {
        line.style.color = '#8f8';
      } else {
        line.style.color = '#aaa';
      }
      line.textContent = `[${type}] ${message}`;
      out.appendChild(line);
    }
    form.onsubmit = async (e) => {
      e.preventDefault();
      append('status', 'Connecting...');
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(protocol + '//' + location.host + '/ws');
      ws.onopen = () => {
        append('status', 'Connected. Starting orchestrator...');
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          append(msg.type, msg.message);
        } catch {
          append('system', ev.data);
        }
      };
      ws.onerror = (ev) => {
        append('error', 'WebSocket error');
      };
      ws.onclose = () => {
        append('status', 'Disconnected.');
      };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket connections and run the orchestrator for the provided goal."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        await websocket.send_json({"type": "status", "message": "Starting Furrow..."})
        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run()
        await websocket.send_json({"type": "status", "message": "Goal complete. Halting."})
    except WebSocketDisconnect:
        try:
            await websocket.send_json({"type": "status", "message": "Disconnected by client."})
        except Exception:
            pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the Uvicorn server for the Furrow web app."""
    uvicorn.run(app, host=host, port=port)
