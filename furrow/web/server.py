from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    max_cycles: int | None = None


async def run_orchestrator_with_ws(goal: str, websocket: WebSocket, model: str | None = None, max_cycles: int | None = None) -> None:
    config = Settings()
    if model:
        config.model = model
    if max_cycles is not None:
        config.max_cycles = max_cycles

    client = LLMClient(config)
    orchestrator = Orchestrator(goal=goal, client=client, config=config)

    # Override console output to send to websocket
    original_print = orchestrator._cycle
    async def cycle_with_ws():
        await original_print()
        # Send state update
        await websocket.send_json({
            "type": "cycle_complete",
            "cycle": orchestrator.cycles,
            "goal": orchestrator.goal,
            "tasks": [t.model_dump() for t in orchestrator._tasks],
        })
    orchestrator._cycle = cycle_with_ws  # type: ignore

    await orchestrator.run()
    await websocket.send_json({"type": "done", "message": "Goal complete"})


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: monospace; max-width: 1000px; margin: 0 auto; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
    h1 { color: #4ec9b0; }
    .panel { border: 1px solid #333; border-radius: 4px; padding: 15px; margin: 10px 0; background: #252526; }
    .panel h3 { margin-top: 0; color: #9cdcfe; }
    .task { padding: 8px; margin: 5px 0; border-radius: 3px; background: #2d2d2d; }
    .task.pending { border-left: 3px solid #cccccc; }
    .task.running { border-left: 3px solid #dcdcaa; }
    .task.completed { border-left: 3px solid #4ec9b0; }
    .task.failed { border-left: 3px solid #f44747; }
    .log { font-size: 12px; color: #999; margin: 2px 0; }
    .log.error { color: #f44747; }
    .log.success { color: #4ec9b0; }
    .log.warning { color: #dcdcaa; }
    input, button { padding: 8px 12px; font-size: 14px; }
    input { width: 60%; background: #333; border: 1px solid #444; color: #d4d4d4; }
    button { background: #0e639c; color: white; border: none; cursor: pointer; }
    button:hover { background: #1177bb; }
    #cycle { font-size: 18px; font-weight: bold; color: #9cdcfe; }
  </style>
</head>
<body>
  <h1>Furrow Web Interface</h1>
  <div class="panel">
    <h3>Start New Goal</h3>
    <form id="form">
      <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
      <button type="submit">Start</button>
    </form>
  </div>
  <div id="status" class="panel" style="display:none;">
    <h3>Status</h3>
    <div id="cycle">Cycle: 0</div>
    <div id="current-goal"></div>
  </div>
  <div id="plan" class="panel" style="display:none;">
    <h3>Plan</h3>
    <pre id="plan-content"></pre>
  </div>
  <div id="tasks" class="panel" style="display:none;">
    <h3>Tasks</h3>
    <div id="tasks-list"></div>
  </div>
  <div id="logs" class="panel" style="display:none;">
    <h3>Logs</h3>
    <div id="logs-content"></div>
  </div>
  <script>
    const form = document.getElementById('form');
    const status = document.getElementById('status');
    const cycleEl = document.getElementById('cycle');
    const goalEl = document.getElementById('current-goal');
    const planEl = document.getElementById('plan');
    const planContent = document.getElementById('plan-content');
    const tasksEl = document.getElementById('tasks');
    const tasksList = document.getElementById('tasks-list');
    const logsEl = document.getElementById('logs');
    const logsContent = document.getElementById('logs-content');

    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      status.style.display = 'block';
      planEl.style.display = 'block';
      tasksEl.style.display = 'block';
      logsEl.style.display = 'block';
      cycleEl.textContent = 'Cycle: 0';
      goalEl.textContent = '';
      planContent.textContent = 'Planning...';
      tasksList.innerHTML = '';
      logsContent.innerHTML = '';

      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        handleMessage(data);
      };
      ws.onclose = () => addLog('Connection closed', 'warning');
      ws.send(JSON.stringify({goal}));
    };

    function handleMessage(data) {
      switch (data.type) {
        case 'cycle_complete':
          cycleEl.textContent = 'Cycle: ' + data.cycle;
          goalEl.textContent = 'Goal: ' + data.goal;
          planContent.textContent = JSON.stringify(data.tasks, null, 2);
          renderTasks(data.tasks);
          break;
        case 'log':
          addLog(data.message, data.level);
          break;
        case 'done':
          addLog(data.message, 'success');
          break;
      }
    }

    function renderTasks(tasks) {
      tasksList.innerHTML = '';
      tasks.forEach(task => {
        const div = document.createElement('div');
        div.className = 'task ' + task.status;
        div.innerHTML = '<strong>' + task.id + ':</strong> ' + task.description + 
          (task.files.length ? ' <span style="color:#999">[' + task.files.join(', ') + ']</span>' : '') +
          (task.result ? '<br><small>' + task.result.substring(0, 200) + '</small>' : '');
        tasksList.appendChild(div);
      });
    }

    function addLog(message, level) {
      const div = document.createElement('div');
      div.className = 'log ' + (level || '');
      div.textContent = message;
      logsContent.appendChild(div);
      logsContent.scrollTop = logsContent.scrollHeight;
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
        max_cycles = data.get("max_cycles")
        await run_orchestrator_with_ws(goal, websocket, model, max_cycles)
    except (WebSocketDisconnect, RuntimeError, OSError):
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
