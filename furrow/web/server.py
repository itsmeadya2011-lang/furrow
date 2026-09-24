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


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
    h1 { color: #4CAF50; }
    .form-group { margin-bottom: 15px; }
    label { display: block; margin-bottom: 5px; font-weight: bold; }
    input, select { width: 100%; padding: 10px; border: 1px solid #333; border-radius: 4px; background: #16213e; color: #eee; font-size: 16px; }
    button { background: #4CAF50; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 16px; }
    button:hover { background: #45a049; }
    button:disabled { background: #666; cursor: not-allowed; }
    #output { background: #0f0f23; border: 1px solid #333; border-radius: 4px; padding: 15px; height: 400px; overflow-y: auto; font-family: monospace; white-space: pre-wrap; }
    .log-line { margin: 2px 0; }
    .log-info { color: #64b5f6; }
    .log-success { color: #81c784; }
    .log-warning { color: #ffb74d; }
    .log-error { color: #e57373; }
    .log-dim { color: #888; }
    .status { display: inline-block; padding: 4px 8px; border-radius: 3px; font-size: 12px; font-weight: bold; margin-left: 10px; }
    .status-running { background: #4CAF50; }
    .status-stopped { background: #f44336; }
    .task-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    .task-table th, .task-table td { padding: 8px; text-align: left; border-bottom: 1px solid #333; }
    .task-table th { background: #16213e; }
  </style>
</head>
<body>
  <h1>Furrow <span id="status" class="status status-stopped">Stopped</span></h1>
  <form id="form" onsubmit="return startFurrow(event)">
    <div class="form-group">
      <label>Goal</label>
      <input id="goal" placeholder="Enter your goal for Furrow (e.g., 'Add JWT auth to the API')" required />
    </div>
    <div class="form-group">
      <label>Model (optional)</label>
      <input id="model" placeholder="Override model (e.g., claude-3-5-sonnet-20241022)" />
    </div>
    <button type="submit" id="startBtn">Start</button>
    <button type="button" onclick="stopFurrow()" id="stopBtn" disabled>Stop</button>
  </form>
  <div id="output"></div>
  <script>
    const output = document.getElementById('output');
    const statusEl = document.getElementById('status');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    let ws = null;

    function appendLog(text, className = '') {
      const line = document.createElement('div');
      line.className = 'log-line ' + className;
      line.textContent = text;
      output.appendChild(line);
      output.scrollTop = output.scrollHeight;
    }

    function setStatus(running) {
      statusEl.textContent = running ? 'Running' : 'Stopped';
      statusEl.className = 'status ' + (running ? 'status-running' : 'status-stopped');
      startBtn.disabled = running;
      stopBtn.disabled = !running;
    }

    async function startFurrow(e) {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      const model = document.getElementById('model').value;
      if (!goal.trim()) return;

      appendLog('Starting Furrow...', 'log-info');
      setStatus(true);

      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'log') {
            appendLog(msg.text, msg.level || '');
          } else if (msg.type === 'tasks') {
            renderTasks(msg.tasks);
          } else if (msg.type === 'done') {
            appendLog('\\n' + msg.text, 'log-success');
            setStatus(false);
          }
        } catch {
          appendLog(ev.data);
        }
      };
      ws.onclose = () => {
        appendLog('\\nConnection closed.', 'log-warning');
        setStatus(false);
      };
      ws.onopen = () => {
        ws.send(JSON.stringify({goal, model}));
      };
    }

    function stopFurrow() {
      if (ws) {
        ws.close();
      }
      setStatus(false);
    }

    function renderTasks(tasks) {
      let html = '<table class="task-table"><tr><th>ID</th><th>Description</th><th>Status</th><th>Files</th></tr>';
      for (const task of tasks) {
        const statusClass = task.status === 'completed' ? 'log-success' : task.status === 'failed' ? 'log-error' : 'log-warning';
        html += `<tr><td>${task.id}</td><td>${task.description}</td><td class="${statusClass}">${task.status}</td><td>${task.files.join(', ')}</td></tr>`;
      }
      html += '</table>';
      // Replace or append task table
      let tableEl = document.getElementById('task-table');
      if (!tableEl) {
        tableEl = document.createElement('div');
        tableEl.id = 'task-table';
        output.appendChild(tableEl);
      }
      tableEl.innerHTML = html;
    }
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        if model:
            from furrow.config import settings
            settings.model = model

        # Custom console that sends to websocket
        class WebSocketConsole:
            def __init__(self, ws: WebSocket):
                self.ws = ws

            async def send(self, type_: str, text: str, level: str = "") -> None:
                await self.ws.send_text(json.dumps({"type": type_, "text": text, "level": level}))

            def print(self, *args, **kwargs):
                text = " ".join(str(a) for a in args)
                level = ""
                if "[green]" in text:
                    level = "log-success"
                elif "[red]" in text:
                    level = "log-error"
                elif "[yellow]" in text:
                    level = "log-warning"
                elif "[dim]" in text:
                    level = "log-dim"
                elif "[bold cyan]" in text:
                    level = "log-info"
                # Strip rich markup
                import re
                text = re.sub(r'\[/?[a-zA-Z ]+\]', '', text)
                asyncio.create_task(self.send("log", text, level))

        ws_console = WebSocketConsole(websocket)

        # Patch console.print temporarily
        from rich.console import Console
        original_print = Console.print

        def ws_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            level = ""
            if "[green]" in text:
                level = "log-success"
            elif "[red]" in text:
                level = "log-error"
            elif "[yellow]" in text:
                level = "log-warning"
            elif "[dim]" in text:
                level = "log-dim"
            elif "[bold cyan]" in text:
                level = "log-info"
            import re
            text = re.sub(r'\[/?[a-zA-Z ]+\]', '', text)
            asyncio.create_task(ws_console.send("log", text, level))

        Console.print = ws_print

        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run()

        await ws_console.send("done", "Goal complete!")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({"type": "log", "text": f"Error: {e}", "level": "log-error"}))
    finally:
        manager.disconnect(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
