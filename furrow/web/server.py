from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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
    body { font-family: monospace; background: #0f0f0f; color: #e0e0e0; margin: 0; padding: 20px; }
    h1 { color: #4CAF50; }
    #form { margin-bottom: 20px; }
    #goal { width: 60%; padding: 8px; font-family: monospace; }
    button { padding: 8px 16px; background: #4CAF50; color: white; border: none; cursor: pointer; }
    button:hover { background: #45a049; }
    #log { background: #1a1a1a; border: 1px solid #333; padding: 15px; border-radius: 6px; white-space: pre-wrap; max-height: 80vh; overflow-y: auto; }
    .plan { color: #2196F3; }
    .task-completed { color: #4CAF50; }
    .task-failed { color: #f44336; }
    .test-passed { color: #4CAF50; }
    .test-failed { color: #f44336; }
    .info { color: #FF9800; }
    .done { color: #4CAF50; font-weight: bold; }
    .start { color: #e0e0e0; }
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
    function append(html) {
      log.innerHTML += html + '\\n';
      log.scrollTop = log.scrollHeight;
    }
    form.onsubmit = async (e) => {
      e.preventDefault();
      append('<span class="start">Starting...</span>');
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'plan') {
            append('<span class="plan">Plan: ' + JSON.stringify(msg.plan.tasks.map(t => t.id + ': ' + t.description)).replace(/"/g, '&quot;') + '</span>');
          } else if (msg.type === 'task_completed') {
            append('<span class="task-completed">Task ' + msg.task_id + ' completed</span>');
          } else if (msg.type === 'task_failed') {
            append('<span class="task-failed">Task ' + msg.task_id + ' failed: ' + msg.error + '</span>');
          } else if (msg.type === 'test_result') {
            const cls = msg.passed ? 'test-passed' : 'test-failed';
            append('<span class="' + cls + '">Tests ' + (msg.passed ? 'passed' : 'failed') + ': ' + msg.summary + '</span>');
          } else if (msg.type === 'done') {
            append('<span class="done">' + msg.message + '</span>');
          } else if (msg.type === 'stop') {
            append('<span class="info">Stopped: ' + msg.reason + '</span>');
          } else if (msg.type === 'no_tasks') {
            append('<span class="info">' + msg.message + '</span>');
          } else {
            append('<span>' + JSON.stringify(msg) + '</span>');
          }
        } catch (err) {
          append('<span>' + ev.data + '</span>');
        }
      };
      ws.onclose = () => append('<span>Session closed.</span>');
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
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
        model = data.get("model")

        queue: deque = deque()

        def emit(event: dict[str, Any]) -> None:
            queue.append(event)

        async def event_sender() -> None:
            while True:
                if queue:
                    event = queue.popleft()
                    await websocket.send_json(event)
                else:
                    await asyncio.sleep(0.05)

        sender_task = asyncio.create_task(event_sender())
        try:
            orchestrator = Orchestrator(goal=goal, event_callback=emit)
            if model:
                from furrow.config import settings
                settings.model = model
            await orchestrator.run()
        finally:
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
