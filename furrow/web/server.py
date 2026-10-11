from __future__ import annotations

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


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


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
    function append(msg) {
      out.textContent += '[' + new Date().toLocaleTimeString() + '] ' + msg + '\\n';
    }
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      const goal = document.getElementById('goal').value;
      append('Goal: ' + goal);
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => ws.send(JSON.stringify({goal}));
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        const data = msg.data || {};
        switch (msg.type) {
          case 'start':
            append('Starting Furrow (max_cycles=' + (data.max_cycles || 0) + ')');
            break;
          case 'plan':
            append('Plan (' + data.rationale + '):');
            (data.tasks || []).forEach(t => append('  [' + t.id + '] ' + t.description));
            break;
          case 'task_created':
            append('Task created: ' + data.id);
            break;
          case 'task_completed':
            append('Task completed: ' + data.id);
            break;
          case 'task_failed':
            append('Task failed: ' + data.id + ': ' + (data.error || 'unknown error'));
            break;
          case 'test':
            append('Tests ' + (data.passed ? 'passed' : 'failed') + ': ' + data.summary);
            break;
          case 'complete':
            append('Goal complete in ' + data.cycles + ' cycles');
            break;
          case 'max_cycles_reached':
            append('Max cycles reached');
            break;
          default:
            append(JSON.stringify(msg));
        }
      };
      ws.onclose = () => append('Closed.');
      ws.onerror = (err) => append('Error: ' + err);
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)

        async def emit(event_type: str, data: dict[str, Any]) -> None:
            await manager.broadcast({"type": event_type, "data": data})

        orchestrator.on_event(emit)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)