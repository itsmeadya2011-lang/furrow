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
    return HTMLResponse(content="""<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f0f23;
      color: #e0e0e0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .container {
      max-width: 960px;
      margin: 0 auto;
      padding: 24px;
    }
    header h1 {
      color: #a78bfa;
      font-size: 28px;
    }
    .card {
      background: #1a1a2e;
      border: 1px solid #2a2a4a;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .card h2 {
      font-size: 14px;
      color: #9ca3af;
      margin-bottom: 10px;
    }
    .form-section input {
      width: 100%;
      padding: 12px;
      background: #0f0f23;
      border: 1px solid #2a2a4a;
      border-radius: 6px;
      color: #e0e0e0;
      font-size: 14px;
      outline: none;
    }
    .form-section input:focus { border-color: #7c3aed; }
    .form-section button {
      margin-top: 10px;
      width: 100%;
      padding: 10px;
      background: #7c3aed;
      color: #fff;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
    }
    .form-section button:hover { background: #8a4adf; }
    .form-section button:disabled { background: #4b5563; cursor: not-allowed; }
    .status-bar {
      display: flex;
      gap: 24px;
      font-size: 13px;
    }
    .status-bar .label { color: #9ca3af; }
    .status-bar .value { font-weight: 600; }
    .status-bar .phase .value { color: #fbbf24; }
    .status-bar .cycle .value { color: #60a5fa; }
    .task-list { list-style: none; }
    .task-item {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      padding: 10px 0;
      border-bottom: 1px solid #2a2a4a;
    }
    .task-item:last-child { border-bottom: none; }
    .task-desc { flex: 1; font-size: 13px; }
    .task-desc strong { color: #93c5fd; }
    .task-result {
      margin-top: 4px;
      font-size: 11px;
      color: #6b7280;
    }
    .badge {
      padding: 2px 10px;
      border-radius: 12px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .badge.pending { background: #374151; color: #d1d5db; }
    .badge.executing { background: #7c2d12; color: #fbbf24; }
    .badge.completed { background: #065f46; color: #6ee7b7; }
    .badge.failed { background: #7f1d1d; color: #fca5a5; }
    .logs pre {
      background: #0d0d1a;
      border: 1px solid #2a2a4a;
      border-radius: 6px;
      padding: 12px;
      color: #9ca3af;
      font-size: 12px;
      max-height: 320px;
      overflow-y: auto;
      white-space: pre-wrap;
      font-family: "SF Mono", Menlo, Consolas, Monaco, monospace;
    }
  </style>
</head>
<body>
  <div class="container">
    <header><h1>Furrow</h1></header>

    <div class="card form-section">
      <form id="form">
        <input id="goal" placeholder="Enter a goal for the agent to work on..." required />
        <button type="submit" id="start-btn">Start</button>
      </form>
    </div>

    <div class="card">
      <div class="status-bar">
        <span class="phase"><span class="label">Phase:</span> <span class="value" id="phase">Idle</span></span>
        <span class="cycle"><span class="label">Cycle:</span> <span class="value" id="cycle">0</span></span>
      </div>
    </div>

    <div class="card">
      <h2>Tasks</h2>
      <ul class="task-list" id="task-list"></ul>
    </div>

    <div class="card">
      <h2>Log</h2>
      <pre id="out"></pre>
    </div>
  </div>

  <script>
    var form = document.getElementById('form');
    var goalInput = document.getElementById('goal');
    var startBtn = document.getElementById('start-btn');
    var phaseEl = document.getElementById('phase');
    var cycleEl = document.getElementById('cycle');
    var taskList = document.getElementById('task-list');
    var out = document.getElementById('out');

    var cycleCount = 0;

    function log(message) {
      var line = document.createElement('div');
      line.textContent = message;
      out.appendChild(line);
      out.scrollTop = out.scrollHeight;
    }

    function handleMessage(msg) {
      log(JSON.stringify(msg));
      switch (msg.type) {
        case 'phase':
          phaseEl.textContent = msg.phase.charAt(0).toUpperCase() + msg.phase.slice(1);
          break;
        case 'plan':
          cycleCount += 1;
          cycleEl.textContent = cycleCount;
          renderTasks(msg.tasks);
          break;
        case 'task_update':
          updateTask(msg.id, msg.status, msg.result);
          break;
        case 'done':
          phaseEl.textContent = 'Complete';
          startBtn.disabled = false;
          break;
      }
    }

    function renderTasks(tasks) {
      taskList.innerHTML = '';
      if (!tasks || tasks.length === 0) {
        var empty = document.createElement('li');
        empty.textContent = 'No tasks planned.';
        empty.style.color = '#6b7280';
        empty.style.fontSize = '12px';
        taskList.appendChild(empty);
        return;
      }
      tasks.forEach(function(t) {
        var li = document.createElement('li');
        li.className = 'task-item';
        li.id = 'task-' + t.id;

        var desc = document.createElement('span');
        desc.className = 'task-desc';
        var title = document.createElement('strong');
        title.textContent = t.id;
        desc.appendChild(title);
        desc.appendChild(document.createTextNode(': ' + t.description));

        var badge = document.createElement('span');
        badge.className = 'badge ' + (t.status || 'pending');
        badge.textContent = t.status || 'pending';

        li.appendChild(desc);
        li.appendChild(badge);
        taskList.appendChild(li);
      });
    }

    function updateTask(id, status, result) {
      var item = document.getElementById('task-' + id);
      if (!item) return;
      var badge = item.querySelector('.badge');
      if (badge) {
        badge.className = 'badge ' + (status || 'pending');
        badge.textContent = status || 'pending';
      }
      if (result) {
        var resultEl = item.querySelector('.task-result');
        if (!resultEl) {
          resultEl = document.createElement('div');
          resultEl.className = 'task-result';
          var desc = item.querySelector('.task-desc');
          desc.appendChild(resultEl);
        }
        resultEl.textContent = result;
      }
    }

    form.onsubmit = function(e) {
      e.preventDefault();
      startBtn.disabled = true;
      phaseEl.textContent = 'Idle';
      cycleEl.textContent = '0';
      cycleCount = 0;
      out.innerHTML = '';
      taskList.innerHTML = '';

      var ws = new WebSocket('ws://' + location.host + '/ws');

      ws.onopen = function() {
        ws.send(JSON.stringify({goal: goalInput.value}));
      };

      ws.onmessage = function(ev) {
        var msg;
        try {
          msg = JSON.parse(ev.data);
        } catch (err) {
          msg = {type: 'log', message: ev.data};
        }
        handleMessage(msg);
      };

      ws.onclose = function() {
        log('Connection closed.');
        startBtn.disabled = false;
      };

      ws.onerror = function() {
        log('WebSocket error: connection lost.');
      };
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

        async def send_event(event: dict) -> None:
            try:
                await websocket.send_json(event)
            except WebSocketDisconnect:
                pass

        orchestrator = Orchestrator(goal=goal, on_event=send_event)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
