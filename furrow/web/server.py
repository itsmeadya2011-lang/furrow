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
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; background: #1a1a2e; color: #eee; }
    h1 { color: #e94560; }
    #log { background: #16213e; border-radius: 8px; padding: 1rem; min-height: 300px; max-height: 70vh; overflow-y: auto; font-family: monospace; font-size: 0.9rem; line-height: 1.5; }
    .event { margin: 0.25rem 0; padding: 0.25rem 0.5rem; border-left: 3px solid #0f3460; }
    .cycle_start { border-color: #e94560; color: #e94560; font-weight: bold; }
    .plan { border-color: #533483; color: #a78bfa; }
    .task_complete { border-color: #16c79a; color: #16c79a; }
    .task_failed { border-color: #e94560; color: #e94560; }
    .test_result { border-color: #f5c518; color: #f5c518; }
    .complete { border-color: #16c79a; color: #16c79a; font-weight: bold; font-size: 1.1rem; }
    input, button { padding: 0.5rem 1rem; font-size: 1rem; border-radius: 4px; border: 1px solid #0f3460; background: #16213e; color: #eee; }
    button { background: #e94560; border: none; cursor: pointer; }
    button:hover { background: #ff6b6b; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <div id="log"></div>
  <script>
    const form = document.getElementById('form');
    const log = document.getElementById('log');
    form.onsubmit = async (e) => {
      e.preventDefault();
      log.innerHTML = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const event = JSON.parse(ev.data);
        const div = document.createElement('div');
        div.className = 'event ' + event.type;
        div.textContent = '[' + event.type + '] ' + JSON.stringify(event);
        log.appendChild(div);
        log.scrollTop = log.scrollHeight;
      };
      ws.onclose = () => {
        const div = document.createElement('div');
        div.className = 'event';
        div.textContent = '[closed] Connection closed.';
        log.appendChild(div);
      };
      ws.onopen = () => ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
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
        orchestrator = Orchestrator(
            goal=goal,
            on_event=lambda e: asyncio.create_task(websocket.send_json(e)),
        )
        await orchestrator.run()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
