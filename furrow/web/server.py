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

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


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
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          out.textContent += msg.text + '\\n';
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
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        async def on_status(event: str, payload: dict) -> None:
            text = _format_event(event, payload)
            await websocket.send_json({"event": event, "text": text})

        orchestrator = Orchestrator(goal=goal, on_status=on_status)
        await orchestrator.run()
        await websocket.send_json({"event": "done", "text": "Session ended."})
    except WebSocketDisconnect:
        pass


def _format_event(event: str, payload: dict) -> str:
    if event == "started":
        return f"Goal: {payload.get('goal', '')}"
    if event == "cycle_start":
        return f"── Cycle {payload.get('cycle', '?')} ──"
    if event == "plan_created":
        return f"Plan: {payload.get('tasks', 0)} tasks. {payload.get('rationale', '')}"
    if event == "no_tasks":
        return "No tasks planned."
    if event == "planning_failed":
        return f"Planning failed: {payload.get('error', '')}"
    if event == "task_completed":
        return f"Task {payload.get('task_id', '?')} completed."
    if event == "task_failed":
        return f"Task {payload.get('task_id', '?')} failed: {payload.get('error', '')}"
    if event == "tests_passed":
        return f"Tests passed: {payload.get('summary', '')}"
    if event == "tests_failed":
        return f"Tests failed: {payload.get('summary', '')}"
    if event == "testing_failed":
        return f"Testing failed: {payload.get('error', '')}"
    if event == "completed":
        return f"Goal complete after {payload.get('cycles', '?')} cycles."
    if event == "max_cycles":
        return f"Stopped after {payload.get('cycles', '?')} cycles."
    return f"[{event}]"


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
