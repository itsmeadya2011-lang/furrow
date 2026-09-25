from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
    h1 { color: #4CAF50; }
    form { margin-bottom: 1rem; }
    input { padding: 0.5rem; width: 70%; }
    button { padding: 0.5rem 1rem; background: #4CAF50; color: white; border: none; cursor: pointer; }
    button:hover { background: #45a049; }
    #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; white-space: pre-wrap; height: 60vh; overflow-y: auto; }
    .cycle { color: #64b5f6; font-weight: bold; }
    .plan { color: #ba68c8; }
    .task { color: #81c784; }
    .error { color: #e57373; }
    .warning { color: #ffb74d; }
    .success { color: #81c784; }
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
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal, websocket=websocket)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
