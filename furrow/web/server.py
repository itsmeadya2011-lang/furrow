from __future__ import annotations

import asyncio

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
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; background: #1e1e1e; color: #d4d4d4; }
    h1 { margin-bottom: 1rem; }
    form { display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
    input[type="text"] { flex: 1; min-width: 300px; padding: 0.5rem; background: #2d2d2d; border: 1px solid #3c3c3c; color: #d4d4d4; border-radius: 4px; }
    button { padding: 0.5rem 1rem; background: #0e639c; color: white; border: none; border-radius: 4px; cursor: pointer; }
    button:disabled { background: #3c3c3c; cursor: not-allowed; }
    #log { background: #121212; border: 1px solid #3c3c3c; border-radius: 4px; padding: 1rem; max-height: 70vh; overflow-y: auto; font-family: monospace; font-size: 0.9rem; line-height: 1.5; }
    .event { margin: 0.25rem 0; padding: 0.25rem 0.5rem; border-radius: 3px; }
    .badge { display: inline-block; width: 10ch; font-weight: bold; font-size: 0.8em; padding: 0 0.4ch; border-radius: 3px; margin-right: 0.5rem; text-align: center; }
    .badge-cycle { background: #00bcd4; color: #121212; }
    .badge-plan { background: #2196f3; color: white; }
    .badge-task_start { background: #ffeb3b; color: #121212; }
    .badge-task_done-completed { background: #4caf50; color: white; }
    .badge-task_done-failed { background: #f44336; color: white; }
    .badge-test-passed { background: #4caf50; color: white; }
    .badge-test-failed { background: #f44336; color: white; }
    .badge-done { background: #4caf50; color: white; font-weight: bold; }
    .badge-error { background: #f44336; color: white; font-weight: bold; }
    .badge-goal { background: #9e9e9e; color: #121212; }
    .task-item { margin-left: 11ch; margin-top: 0.2rem; padding-left: 0.5rem; border-left: 2px solid #3c3c3c; }
    .task-desc { color: #9cdcfe; }
    .task-status { font-weight: bold; }
    .task-status-completed { color: #4caf50; }
    .task-status-failed { color: #f44336; }
    .result { margin-left: 11ch; color: #6a9955; font-size: 0.85em; white-space: pre-wrap; word-break: break-word; }
    .test-summary { margin-left: 11ch; font-weight: bold; }
    .test-failures { margin-left: 11ch; color: #f44336; font-size: 0.85em; }
    .done-reason { margin-left: 11ch; color: #4caf50; font-weight: bold; }
    .error-msg { margin-left: 11ch; color: #f44336; font-weight: bold; }
    .goal-msg { margin-left: 11ch; color: #d4d4d4; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input type="text" id="goal" placeholder="Enter goal" required />
    <button type="submit" id="startBtn">Start</button>
  </form>
  <div id="log"></div>
  <script>
    const form = document.getElementById('form');
    const goalInput = document.getElementById('goal');
    const startBtn = document.getElementById('startBtn');
    const log = document.getElementById('log');

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    function badgeClass(type, subtype) {
      if (subtype) return 'badge-' + type + '-' + subtype;
      return 'badge-' + type;
    }

    function render(data) {
      const div = document.createElement('div');
      div.className = 'event';
      const type = data.type;

      if (type === 'cycle') {
        div.innerHTML = '<span class="badge ' + badgeClass('cycle') + '">CYCLE</span> Cycle ' + escapeHtml(String(data.cycle));
      } else if (type === 'plan') {
        let html = '<span class="badge ' + badgeClass('plan') + '">PLAN</span> ' + escapeHtml(data.rationale || '');
        if (data.tasks && data.tasks.length) {
          html += '<div>';
          data.tasks.forEach(function(t) {
            html += '<div class="task-item"><span class="task-desc">' + escapeHtml(t.id + ': ' + (t.description || '')) + '</span></div>';
          });
          html += '</div>';
        }
        div.innerHTML = html;
      } else if (type === 'task_start') {
        div.innerHTML = '<span class="badge ' + badgeClass('task_start') + '">TASK</span> Starting ' + escapeHtml(data.task_id);
      } else if (type === 'task_done') {
        const status = data.status;
        div.innerHTML = '<span class="badge ' + badgeClass('task_done', status) + '">' + status.toUpperCase() + '</span> ' + escapeHtml(data.task_id);
        if (data.result) {
          const truncated = data.result.length > 200 ? data.result.slice(0, 200) + '...' : data.result;
          div.innerHTML += '<div class="result">' + escapeHtml(truncated) + '</div>';
        }
      } else if (type === 'test') {
        const passed = data.passed;
        div.innerHTML = '<span class="badge ' + badgeClass('test', passed ? 'passed' : 'failed') + '">' + (passed ? 'PASS' : 'FAIL') + '</span> ' + escapeHtml(data.summary || '');
        if (data.failures && data.failures.length) {
          div.innerHTML += '<div class="test-failures">' + data.failures.map(escapeHtml).join('\\n') + '</div>';
        }
      } else if (type === 'done') {
        div.innerHTML = '<span class="badge ' + badgeClass('done') + '">DONE</span><div class="done-reason">' + escapeHtml(data.reason || '') + '</div>';
      } else if (type === 'error') {
        div.innerHTML = '<span class="badge ' + badgeClass('error') + '">ERROR</span><div class="error-msg">' + escapeHtml(data.error || '') + '</div>';
      } else if (type === 'goal') {
        div.innerHTML = '<span class="badge ' + badgeClass('goal') + '">GOAL</span><div class="goal-msg">' + escapeHtml(data.goal || '') + '</div>';
      } else {
        div.innerHTML = '<span class="badge">UNKNOWN</span> ' + escapeHtml(JSON.stringify(data));
      }

      log.appendChild(div);
      log.scrollTop = log.scrollHeight;
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = goalInput.value.trim();
      if (!goal) return;
      startBtn.disabled = true;
      goalInput.disabled = true;
      log.innerHTML = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          render(data);
          if (data.type === 'done' || data.type === 'error') {
            startBtn.disabled = false;
            goalInput.disabled = false;
          }
        } catch (err) {
          console.error(err);
        }
      };
      ws.onclose = () => {
        startBtn.disabled = false;
        goalInput.disabled = false;
      };
      ws.onerror = () => {
        startBtn.disabled = false;
        goalInput.disabled = false;
      };
      ws.send(JSON.stringify({goal: goal, model: null}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    def emit(event: dict) -> None:
        async def _send() -> None:
            try:
                await websocket.send_json(event)
            except Exception:  # noqa: S110, BLE001 - best-effort send to a live socket
                pass
        asyncio.get_running_loop().create_task(_send())

    try:
        data = await websocket.receive_json()
        goal = str(data.get("goal", ""))
        model = data.get("model")
        if model:
            from furrow.config import settings
            settings.model = str(model)
        emit({"type": "goal", "goal": goal})
        orchestrator = Orchestrator(goal=goal, on_event=emit)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001 - report failure to the client
        emit({"type": "error", "error": str(e)})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)