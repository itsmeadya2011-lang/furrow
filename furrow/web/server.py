from __future__ import annotations

import asyncio
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator, StateCallback

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
    body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 1rem; }
    h1 { color: #58a6ff; }
    #log { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; white-space: pre-wrap; max-height: 70vh; overflow-y: auto; }
    .event { margin: 0.25rem 0; }
    .cycle { color: #58a6ff; font-weight: bold; }
    .plan { color: #a5d6ff; }
    .task_completed { color: #3fb950; }
    .task_failed { color: #f85149; }
    .test_passed { color: #3fb950; }
    .test_failed { color: #f85149; }
    .done { color: #3fb950; font-weight: bold; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required style="width: 60%; padding: 0.5rem;" />
    <button type="submit" style="padding: 0.5rem 1rem;">Start</button>
  </form>
  <div id="log"></div>
  <script>
    const form = document.getElementById('form');
    const log = document.getElementById('log');
    function append(cls, text) {
      const div = document.createElement('div');
      div.className = 'event ' + cls;
      div.textContent = text;
      log.appendChild(div);
      log.scrollTop = log.scrollHeight;
    }
    form.onsubmit = async (e) => {
      e.preventDefault();
      log.innerHTML = '';
      const goal = document.getElementById('goal').value;
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => ws.send(JSON.stringify({goal}));
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.event === 'cycle_start') append('cycle', '═══ Cycle ' + msg.payload.cycle + ' ═══');
          else if (msg.event === 'plan') append('plan', 'Plan: ' + JSON.stringify(msg.payload));
          else if (msg.event === 'task_completed') append('task_completed', 'Task ' + msg.payload.id + ' completed');
          else if (msg.event === 'task_failed') append('task_failed', 'Task ' + msg.payload.id + ' failed: ' + msg.payload.error);
          else if (msg.event === 'test_passed') append('test_passed', 'Tests passed: ' + msg.payload.summary);
          else if (msg.event === 'test_failed') append('test_failed', 'Tests failed: ' + msg.payload.summary);
          else if (msg.event === 'done') append('done', 'Goal complete after ' + msg.payload.cycles + ' cycles');
          else if (msg.event === 'max_cycles') append('done', 'Stopped after ' + msg.payload.cycles + ' cycles');
          else append('', msg.text || ev.data);
        } catch {
          append('', ev.data);
        }
      };
      ws.onclose = () => append('', 'Closed.');
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[dict] = asyncio.Queue()

    async def sender() -> None:
        while True:
            msg = await queue.get()
            try:
                await websocket.send_json(msg)
            except Exception:
                break

    sender_task = asyncio.create_task(sender())
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        def emit(event: str, payload: dict) -> None:
            try:
                queue.put_nowait({"event": event, "payload": payload})
            except Exception:
                pass

        orchestrator = Orchestrator(goal=goal, event_callback=emit)
        if model:
            from furrow.config import settings
            settings.model = model
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
