from __future__ import annotations

import asyncio
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator, OrchestratorState

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  h1 { color: #4CAF50; }
  #form { margin-bottom: 1rem; }
  #goal { width: 70%; padding: 0.5rem; font-size: 1rem; }
  button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
  #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; 
         white-space: pre-wrap; font-family: monospace; max-height: 60vh; overflow: auto; }
  .event { margin: 0.25rem 0; }
  .event.start { color: #4CAF50; font-weight: bold; }
  .event.cycle_start { color: #2196F3; border-left: 3px solid #2196F3; padding-left: 0.5rem; }
  .event.plan { color: #FF9800; }
  .event.task_completed { color: #4CAF50; }
  .event.task_failed { color: #f44336; }
  .event.tests_passed { color: #4CAF50; font-weight: bold; }
  .event.tests_failed { color: #f44336; font-weight: bold; }
  .event.complete { color: #4CAF50; font-weight: bold; font-size: 1.2rem; }
  .event.error { color: #f44336; }
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
        const event = JSON.parse(ev.data);
        const div = document.createElement('div');
        div.className = 'event ' + event.type;
        div.textContent = formatEvent(event);
        out.appendChild(div);
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => {
        const div = document.createElement('div');
        div.className = 'event';
        div.textContent = '\\nConnection closed.';
        out.appendChild(div);
      };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };

    function formatEvent(event) {
      switch (event.type) {
        case 'start':
          return `🚀 Starting: ${event.goal}`;
        case 'cycle_start':
          return `\\n═══ Cycle ${event.cycle} ═══\\nGoal: ${event.goal}`;
        case 'planning':
          return `📋 Planning...`;
        case 'plan':
          return `📝 Plan: ${event.plan.rationale}\\n   Tasks: ${event.plan.tasks.length}`;
        case 'tasks':
          return `📦 ${event.tasks.length} tasks queued`;
        case 'executing':
          return `⚡ Executing tasks...`;
        case 'task_completed':
          return `  ✅ Task ${event.task_id} completed`;
        case 'task_failed':
          return `  ❌ Task ${event.task_id} failed: ${event.error}`;
        case 'testing':
          return `🧪 Running tests...`;
        case 'tests_passed':
          return `  ✅ Tests passed: ${event.summary}`;
        case 'tests_failed':
          return `  ❌ Tests failed: ${event.summary}\\n     Failures: ${event.failures.join('; ')}`;
        case 'cycle_end':
          return `--- Cycle ${event.cycle} complete ---`;
        case 'complete':
          return `\\n🎉 Goal complete after ${event.cycles} cycles!`;
        case 'interrupted':
          return `\\n⏹️ Stopped by user`;
        case 'cycle_error':
          return `  💥 Cycle ${event.cycle} error: ${event.error}`;
        default:
          return JSON.stringify(event);
      }
    }
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

        def progress_callback(event: dict) -> None:
            asyncio.create_task(websocket.send_json(event))

        orchestrator = Orchestrator(goal=goal, on_progress=progress_callback)
        if model:
            orchestrator.client.settings.model = model

        await orchestrator.run()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)