from __future__ import annotations

import json
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class ConnectionManager:
    """Manages WebSocket connections for real-time status streaming."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self.active_connections:
            return
        data = json.dumps(message)
        dead: list[WebSocket] = []
        for conn in self.active_connections:
            try:
                await conn.send_text(data)
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.active_connections.remove(conn)


manager = ConnectionManager()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }
    h1 { color: #4CAF50; }
    #events { background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 8px;
              font-family: 'SFMono-Regular', monospace; font-size: 13px; height: 500px; overflow-y: auto;
              white-space: pre-wrap; }
    .event { border-bottom: 1px solid #333; padding: 4px 0; }
    #form { margin-bottom: 20px; }
    input, button { padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
    button { background: #4CAF50; color: white; cursor: pointer; font-weight: bold; }
    button:hover { background: #45a049; }
    button:disabled { background: #999; }
    #status { font-weight: bold; margin-bottom: 10px; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <p>Autonomous coding agent with infinite parallel development loop.</p>
  <div id="status">Idle</div>
  <div id="form">
    <input id="goal" placeholder="Enter your goal..." required style="width: 80%;" />
    <button id="startBtn" type="submit">Start Furrow</button>
  </div>
  <pre id="events"></pre>
  <script>
    const events = document.getElementById('events');
    const statusDiv = document.getElementById('status');
    const startBtn = document.getElementById('startBtn');
    let ws = null;
    function addEvent(data, cls) {
      const ts = new Date().toISOString().slice(11, 19);
      const div = document.createElement('div');
      div.className = 'event ' + (cls || '');
      div.textContent = '[' + ts + '] ' + data;
      events.appendChild(div);
      events.scrollTop = events.scrollHeight;
    }
    document.getElementById('form').onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      if (!goal) return;
      if (ws) ws.close();
      statusDiv.textContent = 'Connecting...';
      startBtn.disabled = true;
      startBtn.textContent = 'Running...';
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => ws.send(JSON.stringify({goal: goal}));
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'status') {
          statusDiv.textContent = msg.message;
        } else if (msg.type === 'done') {
          addEvent(msg.message, msg.cls || 'test-ok');
          statusDiv.textContent = 'Done';
          startBtn.disabled = false;
          startBtn.textContent = 'Start Furrow';
        } else {
          addEvent(msg.message, msg.cls || '');
        }
      };
      ws.onclose = () => {
        statusDiv.textContent = 'Connection closed';
        startBtn.disabled = false;
        startBtn.textContent = 'Start Furrow';
      };
    };
  </script>
</body>
</html>
""",
        headers={"content-type": "text/html"},
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        settings = Settings()
        if model:
            settings.model = model

        from furrow.llm import LLMClient

        client = LLMClient(settings=settings)

        # Set up event broadcasting from orchestrator
        async def emit(event_type: str, message: str, cls: str = "") -> None:
            await manager.broadcast({"type": event_type, "message": message, "cls": cls})

        orchestrator = Orchestrator(goal=goal, client=client, settings=settings, on_event=emit)

        await orchestrator.run()

        done_msg = "Goal complete." if orchestrator._is_done() else "Stopped."
        await manager.broadcast({"type": "done", "message": done_msg, "cls": "test-ok"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await manager.broadcast({"type": "done", "message": f"Error: {e}", "cls": "test-fail"})
    finally:
        manager.disconnect(websocket)
        try:
            await client.close()
        except NameError:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
