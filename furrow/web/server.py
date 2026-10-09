from __future__ import annotations

import asyncio
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, settings
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
    body { font-family: monospace; max-width: 1200px; margin: 0 auto; padding: 20px; }
    h1 { color: #4CAF50; }
    #form { margin-bottom: 20px; }
    #goal { width: 70%; padding: 10px; font-size: 16px; }
    button { padding: 10px 20px; font-size: 16px; background: #4CAF50; color: white; border: none; cursor: pointer; }
    button:hover { background: #45a049; }
    #out { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 5px; max-height: 600px; overflow-y: auto; white-space: pre-wrap; }
    .cycle { border-top: 1px solid #333; margin: 10px 0; padding-top: 10px; }
    .task { margin: 5px 0; padding: 5px 10px; border-radius: 3px; }
    .task.pending { background: #333; }
    .task.running { background: #2d5a2d; }
    .task.completed { background: #1a4d1a; }
    .task.failed { background: #5a1a1a; }
    .plan { background: #1a1a3a; padding: 10px; border-radius: 3px; margin: 10px 0; }
    .test-passed { color: #4CAF50; }
    .test-failed { color: #f44336; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal (e.g., Add JWT auth to the API)" required />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    let ws = null;
    
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      out.textContent += '\\nStarting: ' + goal + '\\n';
      
      if (ws) ws.close();
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        renderMessage(msg);
      };
      ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
      ws.onopen = () => ws.send(JSON.stringify({goal: goal}));
    };
    
    function renderMessage(msg) {
      const div = document.createElement('div');
      if (msg.type === 'cycle') {
        div.className = 'cycle';
        div.textContent = '═══ Cycle ' + msg.cycle + ' ═══';
      } else if (msg.type === 'plan') {
        div.className = 'plan';
        div.textContent = 'Plan: ' + msg.rationale;
        msg.tasks.forEach(t => {
          const tdiv = document.createElement('div');
          tdiv.className = 'task pending';
          tdiv.id = 'task-' + t.id;
          tdiv.textContent = '  [' + t.id + '] ' + t.description + ' (files: ' + (t.files.join(', ') || 'any') + ')';
          div.appendChild(tdiv);
        });
      } else if (msg.type === 'task_start') {
        const tdiv = document.getElementById('task-' + msg.task_id);
        if (tdiv) { tdiv.className = 'task running'; tdiv.textContent = '  [' + msg.task_id + '] ' + msg.description + ' (running...)'; }
      } else if (msg.type === 'task_done') {
        const tdiv = document.getElementById('task-' + msg.task_id);
        if (tdiv) { tdiv.className = 'task ' + msg.status; tdiv.textContent = '  [' + msg.task_id + '] ' + msg.description + ' [' + msg.status + ']'; }
      } else if (msg.type === 'test') {
        div.className = msg.passed ? 'test-passed' : 'test-failed';
        div.textContent = 'Tests: ' + msg.summary;
        if (!msg.passed && msg.failures) {
          msg.failures.forEach(f => {
            const fdiv = document.createElement('div');
            fdiv.textContent = '  ✗ ' + f;
            div.appendChild(fdiv);
          });
        }
      } else if (msg.type === 'log') {
        div.textContent = msg.message;
      } else if (msg.type === 'done') {
        div.style.color = '#4CAF50';
        div.textContent = 'Goal complete!';
      }
      out.appendChild(div);
      out.scrollTop = out.scrollHeight;
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
        
        async def send(msg: dict):
            await websocket.send_text(json.dumps(msg))
        
        orchestrator = Orchestrator(goal=goal, on_progress=send)
        
        await orchestrator.run()
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "log", "message": f"Error: {e}"}))
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
