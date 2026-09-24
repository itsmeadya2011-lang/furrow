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
from furrow.logging import get_logger

app = FastAPI(title="Furrow")
logger = get_logger("furrow.web")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("websocket_connected", total=len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("websocket_disconnected", total=len(self.active_connections))

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; max-width: 800px; margin: 0 auto; padding: 20px; }
  #out { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 5px; max-height: 500px; overflow-y: auto; }
  .log { margin: 2px 0; }
  .log-info { color: #4ec9b0; }
  .log-warning { color: #dcdcaa; }
  .log-error { color: #f44747; }
  .log-success { color: #89d185; }
  input { width: 70%; padding: 8px; }
  button { padding: 8px 16px; background: #007acc; color: white; border: none; border-radius: 3px; cursor: pointer; }
  button:hover { background: #005a9e; }
</style>
</head>
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
          const line = document.createElement('div');
          line.className = 'log log-' + (data.level || 'info');
          line.textContent = '[' + data.timestamp + '] ' + data.message;
          out.appendChild(line);
          out.scrollTop = out.scrollHeight;
        } catch {
          out.textContent += ev.data + '\\n';
        }
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
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
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal:
            await websocket.send_text(json.dumps({"level": "error", "message": "No goal provided", "timestamp": ""}))
            return

        # Create orchestrator with custom logging that sends to websocket
        orchestrator = Orchestrator(goal=goal)

        # Override console printing to send via websocket
        original_print = orchestrator._print_to_websocket
        orchestrator._print_to_websocket = lambda msg: asyncio.create_task(websocket.send_text(json.dumps(msg)))

        await orchestrator.run()
    except WebSocketDisconnect:
        logger.info("websocket_disconnected_during_run")
    except Exception as e:
        logger.error("websocket_error", error=str(e))
        await websocket.send_text(json.dumps({"level": "error", "message": f"Error: {e}", "timestamp": ""}))
    finally:
        manager.disconnect(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
