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

# Store active connections for broadcasting
active_connections: list[WebSocket] = []


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


async def broadcast(message: dict) -> None:
    """Broadcast message to all connected WebSocket clients."""
    dead_connections = []
    for ws in active_connections:
        try:
            await ws.send_json(message)
        except Exception:
            dead_connections.append(ws)
    for ws in dead_connections:
        active_connections.remove(ws)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 1200px; margin: 0 auto; padding: 20px; }
        #form { margin-bottom: 20px; }
        #goal { width: 70%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        #status { padding: 10px; background: #f5f5f5; border-radius: 4px; margin-bottom: 20px; }
        #tasks { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
        .task { padding: 10px; border-radius: 4px; min-width: 200px; flex: 1; }
        .task.pending { background: #fff3cd; }
        .task.running { background: #cce5ff; }
        .task.completed { background: #d4edda; }
        .task.failed { background: #f8d7da; }
        #log { background: #1e1e1e; color: #d4d4d4; padding: 15px; height: 400px; overflow-y: auto; border-radius: 4px; white-space: pre-wrap; }
        .log-line { margin: 2px 0; }
        .log-info { color: #9cdcfe; }
        .log-success { color: #4ec9b0; }
        .log-error { color: #f14c4c; }
        .log-warning { color: #dcdcaa; }
    </style>
</head>
<body>
    <h1>Furrow - Autonomous Development Loop</h1>
    <div id="form">
        <input id="goal" placeholder="Enter goal (e.g., 'Add JWT authentication to the API')" required />
        <button type="submit" onclick="startFurrow()">Start</button>
    </div>
    <div id="status">Waiting for goal...</div>
    <div id="tasks"></div>
    <div id="log"></div>
    <script>
        let ws;
        function startFurrow() {
            const goal = document.getElementById('goal').value;
            if (!goal) return;
            
            document.getElementById('status').textContent = 'Connecting...';
            document.getElementById('log').textContent = '';
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                const msg = JSON.parse(ev.data);
                handleMessage(msg);
            };
            ws.onclose = () => {
                document.getElementById('status').textContent = 'Disconnected';
            };
            ws.onopen = () => {
                ws.send(JSON.stringify({type: 'start', goal: goal}));
                document.getElementById('status').textContent = 'Running...';
            };
        }
        
        function handleMessage(msg) {
            const log = document.getElementById('log');
            const tasksDiv = document.getElementById('tasks');
            const statusDiv = document.getElementById('status');
            
            switch (msg.type) {
                case 'cycle_start':
                    statusDiv.textContent = `Cycle ${msg.cycle} - Planning...`;
                    break;
                case 'plan':
                    statusDiv.textContent = `Cycle ${msg.cycle} - Executing ${msg.tasks.length} tasks...`;
                    renderTasks(msg.tasks);
                    break;
                case 'task_update':
                    updateTask(msg.task);
                    break;
                case 'test_result':
                    statusDiv.textContent = `Cycle ${msg.cycle} - Tests ${msg.passed ? 'passed' : 'failed'}`;
                    logMessage(msg.passed ? 'success' : 'error', msg.summary);
                    if (msg.failures && msg.failures.length > 0) {
                        msg.failures.forEach(f => logMessage('error', '  ' + f));
                    }
                    break;
                case 'cycle_complete':
                    statusDiv.textContent = `Cycle ${msg.cycle} complete`;
                    break;
                case 'done':
                    statusDiv.textContent = 'Goal complete!';
                    logMessage('success', 'Goal completed successfully');
                    break;
                case 'error':
                    logMessage('error', msg.message);
                    break;
                case 'log':
                    logMessage(msg.level, msg.message);
                    break;
            }
        }
        
        function renderTasks(tasks) {
            const tasksDiv = document.getElementById('tasks');
            tasksDiv.innerHTML = '';
            tasks.forEach(task => {
                const div = document.createElement('div');
                div.className = `task ${task.status}`;
                div.id = `task-${task.id}`;
                div.innerHTML = `<strong>${task.id}: ${task.description}</strong><br><small>${task.files.join(', ') || 'any'}</small>`;
                tasksDiv.appendChild(div);
            });
        }
        
        function updateTask(task) {
            const div = document.getElementById(`task-${task.id}`);
            if (div) {
                div.className = `task ${task.status}`;
                div.innerHTML = `<strong>${task.id}: ${task.description}</strong><br><small>${task.files.join(', ') || 'any'}</small><br><em>${task.result || ''}</em>`;
            }
        }
        
        function logMessage(level, message) {
            const log = document.getElementById('log');
            const line = document.createElement('div');
            line.className = `log-line log-${level}`;
            line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            log.appendChild(line);
            log.scrollTop = log.scrollHeight;
        }
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "start":
                goal = data.get("goal", "")
                await run_orchestrator(goal, websocket)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


async def run_orchestrator(goal: str, websocket: WebSocket) -> None:
    """Run the orchestrator with WebSocket updates."""
    client = None
    try:
        from furrow.llm import LLMClient
        client = LLMClient()
        orchestrator = Orchestrator(goal=goal, client=client)
        
        # We need to hook into the orchestrator to send updates
        # For now, run it and send periodic updates
        await websocket.send_json({"type": "cycle_start", "cycle": 1})
        
        # Run a modified cycle that sends updates
        # This is a simplified version - in production you'd want to properly hook into the orchestrator
        while True:
            # Simulate cycle
            await asyncio.sleep(1)
            await websocket.send_json({"type": "cycle_complete", "cycle": orchestrator.cycles})
            if orchestrator._is_done():
                await websocket.send_json({"type": "done"})
                break
                
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        if client:
            # Cleanup if needed
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
