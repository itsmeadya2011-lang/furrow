from __future__ import annotations

import asyncio
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import settings
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
  <meta charset="utf-8" />
  <title>Furrow</title>
  <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 2rem; }
    input { padding: 0.4rem; margin-right: 0.4rem; }
    #events { list-style: none; padding-left: 0; margin-top: 1.5rem; }
    #events li { padding: 0.25rem 0.5rem; border-left: 3px solid #ccc; margin-bottom: 0.25rem; }
    #events li .type { font-weight: bold; margin-right: 0.5rem; }
    .ok { border-left-color: #2b8a3e; }
    .bad { border-left-color: #c92a2a; }
    .info { border-left-color: #1971c2; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <input id="model" placeholder="Model (optional)" />
    <button type="submit">Start</button>
  </form>
  <ul id="events"></ul>
  <script>
    const form = document.getElementById('form');
    const list = document.getElementById('events');

    function cls(event) {
      if (event.type === 'task_complete' || event.type === 'cycle_complete') return 'ok';
      if (event.type === 'task_failed' || event.type === 'task_blocked') return 'bad';
      if (event.type === 'test_result') return event.passed ? 'ok' : 'bad';
      return 'info';
    }

    function summary(event) {
      const ids = Array.isArray(event.task_ids) ? event.task_ids.join(', ') : '';
      switch (event.type) {
        case 'cycle_start': return 'Cycle ' + event.cycle + ': ' + (event.goal || '');
        case 'plan': return (event.rationale || 'plan') + ' [' + ids + ']';
        case 'task_start': return event.task_id + ': ' + (event.description || '');
        case 'task_complete': return event.task_id + ' completed';
        case 'task_failed': return event.task_id + ' failed: ' + (event.error || '');
        case 'task_blocked': return event.task_id + ' blocked by ' + (event.blocked_by || []).join(', ');
        case 'test_result':
          return (event.passed ? 'passed' : 'failed') + ': ' + (event.summary || '');
        case 'cycle_complete': return 'Cycle ' + event.cycle + ' complete';
        case 'halted': return 'Halted (' + (event.reason || '') + ')';
        case 'closed': return 'Stream closed';
        default: return JSON.stringify(event);
      }
    }

    function add(event) {
      const li = document.createElement('li');
      li.className = cls(event);
      const label = document.createElement('span');
      label.className = 'type';
      label.textContent = event.type;
      li.appendChild(label);
      li.appendChild(document.createTextNode(summary(event)));
      list.appendChild(li);
    }

    form.onsubmit = (e) => {
      e.preventDefault();
      list.textContent = '';
      const payload = {goal: document.getElementById('goal').value};
      const model = document.getElementById('model').value.trim();
      if (model) payload.model = model;
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => add(JSON.parse(ev.data));
      ws.onclose = () => add({type: 'closed'});
      ws.onopen = () => ws.send(JSON.stringify(payload));
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
        if model:
            settings.model = model
            settings.planner_model = model
            settings.worker_model = model
            settings.tester_model = model

        pending: set[asyncio.Task] = set()

        async def on_event(event: dict[str, Any]) -> None:
            try:
                await websocket.send_json(event)
            except Exception:
                return

        def dispatch(event: dict[str, Any]) -> None:
            task = asyncio.create_task(on_event(event))
            pending.add(task)
            task.add_done_callback(pending.discard)

        orchestrator = Orchestrator(goal=goal, on_event=dispatch)
        await orchestrator.run()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await websocket.send_json({"type": "closed"})
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
