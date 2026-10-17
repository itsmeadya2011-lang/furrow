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


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str) -> None:
        dead: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)


manager = ConnectionManager()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <div id="status" style="color:red">disconnected</div>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const status = document.getElementById('status');
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    let ws = null;

    function stamp() {
      return new Date().toLocaleTimeString();
    }

    function log(text) {
      out.textContent += '[' + stamp() + '] ' + text + '\\n';
    }

    function connect() {
      if (ws && ws.readyState === WebSocket.OPEN) return;
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => {
        status.textContent = 'connected';
        status.style.color = 'green';
        log('connection established');
      };
      ws.onmessage = (ev) => log(ev.data);
      ws.onclose = () => {
        status.textContent = 'disconnected';
        status.style.color = 'red';
        log('connection closed');
        ws = null;
      };
      ws.onerror = () => log('connection error');
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      connect();
      log('goal: ' + document.getElementById('goal').value);
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        await manager.broadcast("connection established")
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        await manager.broadcast("goal: " + goal)
        await manager.broadcast("starting orchestrator")
        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run()
        await manager.broadcast("completed")
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
