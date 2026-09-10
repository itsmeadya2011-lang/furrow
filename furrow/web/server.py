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
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #00d9ff; }
        #form { margin-bottom: 20px; }
        #goal { width: 70%; padding: 10px; font-size: 16px; background: #16213e; border: 1px solid #0f3460; color: #eee; }
        button { padding: 10px 20px; font-size: 16px; background: #0f3460; color: #00d9ff; border: 1px solid #00d9ff; cursor: pointer; }
        button:hover { background: #00d9ff; color: #1a1a2e; }
        #out { background: #16213e; padding: 15px; border-radius: 5px; max-height: 60vh; overflow-y: auto; white-space: pre-wrap; }
        .cycle { border-left: 3px solid #00d9ff; padding-left: 10px; margin: 10px 0; }
        .task { margin-left: 20px; }
        .task.completed { color: #00ff88; }
        .task.failed { color: #ff4444; }
        .task.running { color: #ffaa00; }
        .test.passed { color: #00ff88; }
        .test.failed { color: #ff4444; }
        .plan { background: #0f3460; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
    <button type="submit">Start</button>
  </form>
  <div id="out"></div>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    let ws = null;
    
    function append(msg, className = '') {
      const div = document.createElement('div');
      div.className = className;
      div.textContent = msg;
      out.appendChild(div);
      out.scrollTop = out.scrollHeight;
    }
    
    function formatEvent(event) {
      switch(event.type) {
        case 'start':
          return `🚀 Starting: ${event.goal}`;
        case 'cycle_start':
          return `\\n═══ Cycle ${event.cycle} ═══`;
        case 'planning_start':
          return `  📋 Planning...`;
        case 'planning_complete':
          const tasks = event.plan.tasks.map(t => `    • ${t.id}: ${t.description} (${t.files.join(', ') || 'any'})`).join('\\n');
          return `  📋 Plan:\\n${tasks}\\n  Rationale: ${event.plan.rationale}`;
        case 'task_start':
          return `  ▶ Task ${event.task_id} started (attempt ${event.attempt})`;
        case 'task_retry':
          return `  ↻ Task ${event.task_id} retry ${event.attempt}/${event.max_retries}`;
        case 'task_complete':
          return `  ✓ Task ${event.task_id} completed: ${event.result.substring(0, 100)}...`;
        case 'task_error':
          return `  ✗ Task ${event.task_id} error: ${event.error}`;
        case 'task_failed':
          return `  ✗ Task ${event.task_id} failed after ${event.max_retries} attempts: ${event.error}`;
        case 'testing_start':
          return `  🧪 Running tests...`;
        case 'tests_passed':
          return `  ✓ Tests passed: ${event.summary}`;
        case 'tests_failed':
          return `  ✗ Tests failed: ${event.summary}\\n    Failures: ${event.failures.join('; ')}`;
        case 'complete':
          return `\\n✅ Goal complete after ${event.cycles} cycles!`;
        case 'max_cycles_reached':
          return `\\n⚠️ Max cycles (${event.max_cycles}) reached.`;
        case 'no_tasks':
          return `  ℹ️ No tasks planned. Goal may be complete.`;
        default:
          return JSON.stringify(event);
      }
    }
    
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      if (!goal) return;
      
      out.innerHTML = '';
      append(`Starting: ${goal}`);
      
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const event = JSON.parse(ev.data);
          append(formatEvent(event), event.type);
        } catch {
          append(ev.data);
        }
      };
      ws.onclose = () => append('\\n🔌 Connection closed.');
      ws.onerror = (err) => append('\\n❌ WebSocket error');
      ws.send(JSON.stringify({goal}));
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
        
        async def progress_callback(event: dict) -> None:
            await websocket.send_text(json.dumps(event))
        
        orchestrator = Orchestrator(goal=goal, progress_callback=progress_callback)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
