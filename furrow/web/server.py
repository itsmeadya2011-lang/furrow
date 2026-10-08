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

app = FastAPI(title="Furrow Web UI")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None
    max_cycles: Optional[int] = None


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


# Custom console that broadcasts to WebSocket
class WebConsole:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
    
    def print(self, *args, **kwargs) -> None:
        # Convert rich markup to plain text for web
        import re
        text = " ".join(str(arg) for arg in args)
        # Strip rich markup
        text = re.sub(r"\[/?[^\]]+\]", "", text)
        asyncio.create_task(self.websocket.send_json({"type": "log", "content": text}))


# Patch console for web
import furrow.core.orchestrator as orchestrator_module
original_console = orchestrator_module.console


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow - Autonomous Coding Agent</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #eaeaea; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #4CAF50; margin-bottom: 10px; }
        .subtitle { color: #888; margin-bottom: 30px; }
        
        .panel { background: #16213e; border-radius: 8px; padding: 20px; margin-bottom: 20px; border: 1px solid #0f3460; }
        .panel h2 { margin-top: 0; color: #4CAF50; border-bottom: 1px solid #0f3460; padding-bottom: 10px; }
        
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 10px; border: 1px solid #0f3460; border-radius: 4px; background: #0f0f23; color: #eaeaea; font-size: 14px; }
        input:focus, select:focus { outline: none; border-color: #4CAF50; }
        button { background: #4CAF50; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 16px; font-weight: bold; }
        button:hover { background: #45a049; }
        button:disabled { background: #666; cursor: not-allowed; }
        button.secondary { background: #0f3460; }
        button.secondary:hover { background: #1a4a7a; }
        
        .status { padding: 10px; border-radius: 4px; margin-bottom: 20px; }
        .status.running { background: #1b4d3e; border: 1px solid #4CAF50; }
        .status.stopped { background: #4d1b1b; border: 1px solid #f44336; }
        .status.idle { background: #1a1a2e; border: 1px solid #444; }
        
        .progress-bar { height: 8px; background: #0f3460; border-radius: 4px; overflow: hidden; margin: 10px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); transition: width 0.3s ease; }
        
        .cycle { background: #0f0f23; border-radius: 4px; padding: 15px; margin-bottom: 10px; border-left: 3px solid #4CAF50; }
        .cycle-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
        .cycle-number { font-weight: bold; color: #4CAF50; }
        .cycle-time { color: #888; font-size: 12px; }
        
        .task { background: #16213e; border-radius: 4px; padding: 10px; margin: 8px 0; border-left: 3px solid #2196F3; }
        .task.completed { border-left-color: #4CAF50; }
        .task.failed { border-left-color: #f44336; }
        .task.pending { border-left-color: #FFC107; }
        .task-header { display: flex; justify-content: space-between; margin-bottom: 5px; }
        .task-id { font-family: monospace; font-size: 12px; color: #888; }
        .task-desc { font-weight: bold; }
        .task-files { font-size: 12px; color: #888; }
        .task-result { margin-top: 8px; padding: 8px; background: #0f0f23; border-radius: 4px; font-family: monospace; font-size: 12px; white-space: pre-wrap; max-height: 200px; overflow: auto; }
        
        .log { background: #0f0f23; border-radius: 4px; padding: 15px; max-height: 400px; overflow: auto; font-family: monospace; font-size: 12px; line-height: 1.5; }
        .log-line { margin: 2px 0; }
        .log-line.info { color: #2196F3; }
        .log-line.success { color: #4CAF50; }
        .log-line.warning { color: #FFC107; }
        .log-line.error { color: #f44336; }
        .log-line.system { color: #9C27B0; }
        
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat { background: #16213e; border-radius: 8px; padding: 15px; text-align: center; }
        .stat-value { font-size: 32px; font-weight: bold; color: #4CAF50; }
        .stat-label { color: #888; font-size: 14px; margin-top: 5px; }
        
        .tabs { display: flex; border-bottom: 1px solid #0f3460; margin-bottom: 20px; }
        .tab { padding: 10px 20px; cursor: pointer; color: #888; border-bottom: 2px solid transparent; }
        .tab.active { color: #4CAF50; border-bottom-color: #4CAF50; }
        .tab:hover { color: #eaeaea; }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌾 Furrow</h1>
        <p class="subtitle">Autonomous coding agent with infinite parallel development loop</p>
        
        <div class="stats" id="stats">
            <div class="stat"><div class="stat-value" id="stat-cycles">0</div><div class="stat-label">Cycles</div></div>
            <div class="stat"><div class="stat-value" id="stat-total">0</div><div class="stat-label">Total Tasks</div></div>
            <div class="stat"><div class="stat-value" id="stat-completed">0</div><div class="stat-label">Completed</div></div>
            <div class="stat"><div class="stat-value" id="stat-failed">0</div><div class="stat-label">Failed</div></div>
        </div>
        
        <div class="panel">
            <h2>Start New Session</h2>
            <div class="form-group">
                <label for="goal">Goal</label>
                <input type="text" id="goal" placeholder="e.g., Add JWT authentication to the API" required />
            </div>
            <div class="form-group">
                <label for="model">Model (optional)</label>
                <input type="text" id="model" placeholder="e.g., claude-3-5-sonnet-20241022" />
            </div>
            <div class="form-group">
                <label for="maxCycles">Max Cycles (0 = unlimited)</label>
                <input type="number" id="maxCycles" value="0" min="0" />
            </div>
            <button id="startBtn" onclick="startSession()">Start Furrow</button>
            <button class="secondary" id="stopBtn" onclick="stopSession()" disabled>Stop</button>
            <button class="secondary" onclick="clearState()">Clear State</button>
        </div>
        
        <div class="status idle" id="status">
            <strong>Status:</strong> <span id="statusText">Idle</span>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progressFill" style="width: 0%"></div>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="showTab('cycles')">Cycles</div>
            <div class="tab" onclick="showTab('tasks')">Tasks</div>
            <div class="tab" onclick="showTab('log')">Log</div>
        </div>
        
        <div id="cyclesTab" class="tab-content active">
            <div id="cyclesContainer"></div>
        </div>
        
        <div id="tasksTab" class="tab-content">
            <div id="tasksContainer"></div>
        </div>
        
        <div id="logTab" class="tab-content">
            <div class="log" id="logContainer"></div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let isRunning = false;
        
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName + 'Tab').classList.add('active');
        }
        
        function connectWebSocket() {
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onopen = () => console.log('WebSocket connected');
            ws.onmessage = handleMessage;
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                setTimeout(connectWebSocket, 1000);
            };
            ws.onerror = (err) => console.error('WebSocket error:', err);
        }
        
        function handleMessage(event) {
            const msg = JSON.parse(event.data);
            switch (msg.type) {
                case 'status':
                    updateStatus(msg);
                    break;
                case 'cycle_start':
                    addCycle(msg);
                    break;
                case 'cycle_complete':
                    completeCycle(msg);
                    break;
                case 'task_update':
                    updateTask(msg);
                    break;
                case 'test_result':
                    showTestResult(msg);
                    break;
                case 'log':
                    addLog(msg.content, msg.level || 'info');
                    break;
                case 'progress':
                    updateProgress(msg);
                    break;
                case 'done':
                    sessionDone(msg);
                    break;
            }
        }
        
        function startSession() {
            const goal = document.getElementById('goal').value.trim();
            if (!goal) return alert('Please enter a goal');
            
            const model = document.getElementById('model').value.trim() || null;
            const maxCycles = parseInt(document.getElementById('maxCycles').value) || 0;
            
            isRunning = true;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('goal').disabled = true;
            document.getElementById('model').disabled = true;
            document.getElementById('maxCycles').disabled = true;
            
            document.getElementById('cyclesContainer').innerHTML = '';
            document.getElementById('tasksContainer').innerHTML = '';
            document.getElementById('logContainer').innerHTML = '';
            updateStatus({status: 'running', cycles: 0, totalTasks: 0, completed: 0, failed: 0});
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({goal, model, maxCycles}));
            } else {
                connectWebSocket();
                setTimeout(() => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({goal, model, maxCycles}));
                    }
                }, 500);
            }
        }
        
        function stopSession() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({action: 'stop'}));
            }
            isRunning = false;
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('goal').disabled = false;
            document.getElementById('model').disabled = false;
            document.getElementById('maxCycles').disabled = false;
        }
        
        function clearState() {
            fetch('/api/state', {method: 'DELETE'}).then(() => {
                document.getElementById('cyclesContainer').innerHTML = '';
                document.getElementById('tasksContainer').innerHTML = '';
                updateStatus({status: 'idle', cycles: 0, totalTasks: 0, completed: 0, failed: 0});
            });
        }
        
        function updateStatus(data) {
            const statusEl = document.getElementById('status');
            const statusText = document.getElementById('statusText');
            
            if (data.status === 'running') {
                statusEl.className = 'status running';
                statusText.textContent = 'Running...';
            } else if (data.status === 'done') {
                statusEl.className = 'status stopped';
                statusText.textContent = 'Completed';
            } else {
                statusEl.className = 'status idle';
                statusText.textContent = 'Idle';
            }
            
            document.getElementById('stat-cycles').textContent = data.cycles || 0;
            document.getElementById('stat-total').textContent = data.totalTasks || 0;
            document.getElementById('stat-completed').textContent = data.completed || 0;
            document.getElementById('stat-failed').textContent = data.failed || 0;
        }
        
        function updateProgress(data) {
            const fill = document.getElementById('progressFill');
            if (data.total > 0) {
                fill.style.width = (data.completed / data.total * 100) + '%';
            }
        }
        
        function addCycle(data) {
            const container = document.getElementById('cyclesContainer');
            const cycleDiv = document.createElement('div');
            cycleDiv.className = 'cycle';
            cycleDiv.id = 'cycle-' + data.cycle;
            cycleDiv.innerHTML = `
                <div class="cycle-header">
                    <span class="cycle-number">Cycle ${data.cycle}</span>
                    <span class="cycle-time">${new Date().toLocaleTimeString()}</span>
                </div>
                <div class="cycle-plan">Planning...</div>
            `;
            container.insertBefore(cycleDiv, container.firstChild);
        }
        
        function completeCycle(data) {
            const cycleDiv = document.getElementById('cycle-' + data.cycle);
            if (cycleDiv) {
                cycleDiv.querySelector('.cycle-plan').innerHTML = 
                    '<strong>Plan:</strong> ' + data.plan.rationale + 
                    '<br><strong>Tasks:</strong> ' + data.plan.tasks.length;
            }
        }
        
        function updateTask(data) {
            let taskDiv = document.getElementById('task-' + data.taskId);
            if (!taskDiv) {
                taskDiv = document.createElement('div');
                taskDiv.className = 'task ' + data.status;
                taskDiv.id = 'task-' + data.taskId;
                document.getElementById('tasksContainer').insertBefore(taskDiv, document.getElementById('tasksContainer').firstChild);
            }
            
            taskDiv.className = 'task ' + data.status;
            taskDiv.innerHTML = `
                <div class="task-header">
                    <span class="task-desc">${data.description}</span>
                    <span class="task-id">${data.taskId}</span>
                </div>
                <div class="task-files">Files: ${data.files.join(', ') || 'any'}</div>
                ${data.result ? '<div class="task-result">' + escapeHtml(data.result) + '</div>' : ''}
            `;
        }
        
        function showTestResult(data) {
            addLog('Test Result: ' + (data.passed ? 'PASSED' : 'FAILED'), data.passed ? 'success' : 'error');
            if (data.summary) addLog(data.summary, 'info');
            data.failures.forEach(f => addLog('  - ' + f, 'error'));
        }
        
        function addLog(text, level = 'info') {
            const container = document.getElementById('logContainer');
            const line = document.createElement('div');
            line.className = 'log-line ' + level;
            line.textContent = '[' + new Date().toLocaleTimeString() + '] ' + text;
            container.appendChild(line);
            container.scrollTop = container.scrollHeight;
        }
        
        function sessionDone(data) {
            isRunning = false;
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('goal').disabled = false;
            document.getElementById('model').disabled = false;
            document.getElementById('maxCycles').disabled = false;
            
            updateStatus({status: 'done', cycles: data.cycles, totalTasks: data.totalTasks, completed: data.completed, failed: data.failed});
            addLog('Session completed. ' + (data.failed > 0 ? 'Some tasks failed.' : 'All tasks completed successfully!'), data.failed > 0 ? 'error' : 'success');
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Connect on load
        connectWebSocket();
    </script>
</body>
</html>
""")


@app.get("/api/state")
async def get_state() -> dict:
    """Get current orchestrator state."""
    state_file = Path(settings.state_file)
    if not state_file.exists():
        return {"goal": "", "cycles": 0, "all_tasks": [], "completed_tasks": [], "failed_tasks": []}
    
    try:
        with open(state_file) as f:
            return json.load(f)
    except Exception:
        return {"goal": "", "cycles": 0, "all_tasks": [], "completed_tasks": [], "failed_tasks": []}


@app.delete("/api/state")
async def clear_state() -> dict:
    """Clear orchestrator state."""
    state_file = Path(settings.state_file)
    if state_file.exists():
        state_file.unlink()
    return {"success": True}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        # Send initial state
        state = await get_state()
        await websocket.send_json({"type": "status", **state})
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("action") == "stop":
                await websocket.send_json({"type": "done", "message": "Stopped by user"})
                break
            
            goal = data.get("goal", "")
            model = data.get("model")
            max_cycles = data.get("maxCycles", 0) or 0
            
            if not goal:
                await websocket.send_json({"type": "error", "message": "Goal is required"})
                continue
            
            # Override model if provided
            if model:
                settings.model = model
            
            # Run orchestrator with WebSocket updates
            orchestrator = Orchestrator(goal=goal, max_cycles=max_cycles)
            
            # Monkey-patch console for this session
            web_console = WebConsole(websocket)
            orchestrator_module.console = web_console
            
            try:
                await orchestrator.run()
                
                # Send final status
                summary = orchestrator.get_progress_summary()
                await websocket.send_json({
                    "type": "done",
                    "cycles": summary["cycles"],
                    "totalTasks": summary["total_tasks"],
                    "completed": summary["completed"],
                    "failed": summary["failed"],
                })
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
            finally:
                orchestrator_module.console = original_console
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        manager.disconnect(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


from pathlib import Path