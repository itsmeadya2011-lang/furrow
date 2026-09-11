from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, State, TaskModel
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Store active orchestrator for status updates
active_orchestrator: Orchestrator | None = None
connected_websockets: list[WebSocket] = []


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None
    resume: bool = False


async def broadcast_state_update() -> None:
    """Broadcast current state to all connected websockets."""
    if not active_orchestrator:
        return
    
    state_data = {
        "type": "state_update",
        "goal": active_orchestrator.goal,
        "current_cycle": active_orchestrator.state.current_cycle,
        "completed_cycles": active_orchestrator.state.completed_cycles,
        "tasks": [
            {
                "id": t.id,
                "description": t.description,
                "files": t.files,
                "dependencies": t.dependencies,
                "status": t.status,
                "result": t.result,
                "cycle": t.cycle,
            }
            for t in active_orchestrator.state.tasks
        ],
        "timestamp": datetime.now().isoformat(),
    }
    
    message = json.dumps(state_data)
    disconnected = []
    for ws in connected_websockets:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    
    for ws in disconnected:
        connected_websockets.remove(ws)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow - Autonomous Coding Agent</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #4CAF50; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 10px; border: 1px solid #333; background: #16213e; color: #eee; border-radius: 4px; }
        button { padding: 12px 24px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        button:disabled { background: #666; cursor: not-allowed; }
        .status { padding: 15px; border-radius: 4px; margin: 15px 0; }
        .status.running { background: #16213e; border-left: 4px solid #4CAF50; }
        .status.complete { background: #16213e; border-left: 4px solid #2196F3; }
        .status.error { background: #16213e; border-left: 4px solid #f44336; }
        .cycle { margin: 10px 0; padding: 15px; background: #16213e; border-radius: 4px; border-left: 4px solid #2196F3; }
        .task { margin: 5px 0; padding: 10px; background: #0f0f1a; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; }
        .task.pending { border-left: 4px solid #ff9800; }
        .task.completed { border-left: 4px solid #4CAF50; }
        .task.failed { border-left: 4px solid #f44336; }
        .task-info { flex: 1; }
        .task-id { font-weight: bold; color: #4CAF50; }
        .task-desc { margin: 5px 0; }
        .task-files { font-size: 12px; color: #888; }
        .task-status { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; text-transform: uppercase; }
        .task-status.pending { background: #ff9800; color: #000; }
        .task-status.completed { background: #4CAF50; color: white; }
        .task-status.failed { background: #f44336; color: white; }
        .log { background: #0f0f1a; padding: 15px; border-radius: 4px; max-height: 400px; overflow-y: auto; font-family: monospace; font-size: 13px; }
        .log-entry { margin: 2px 0; }
        .log-entry.info { color: #2196F3; }
        .log-entry.success { color: #4CAF50; }
        .log-entry.error { color: #f44336; }
        .log-entry.warning { color: #ff9800; }
    </style>
</head>
<body>
    <h1>Furrow - Autonomous Coding Agent</h1>
    
    <div class="form-group">
        <label for="goal">Goal</label>
        <input type="text" id="goal" placeholder="Enter your goal (e.g., 'Add JWT auth to the API')" />
    </div>
    
    <div class="form-group">
        <label for="model">Model (optional)</label>
        <input type="text" id="model" placeholder="e.g., claude-3-5-sonnet-20241022" />
    </div>
    
    <div class="form-group">
        <label>
            <input type="checkbox" id="resume" /> Resume from saved state
        </label>
    </div>
    
    <button id="startBtn" onclick="startFurrow()">Start Furrow</button>
    <button id="stopBtn" onclick="stopFurrow()" disabled>Stop</button>
    
    <div id="status" class="status" style="display: none;"></div>
    
    <h2>Cycles</h2>
    <div id="cycles"></div>
    
    <h2>Task Log</h2>
    <div id="log" class="log"></div>

    <script>
        let ws = null;
        let isRunning = false;
        
        function addLog(message, type = 'info') {
            const log = document.getElementById('log');
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + type;
            entry.textContent = '[' + new Date().toLocaleTimeString() + '] ' + message;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
        }
        
        function updateStatus(message, type = 'running') {
            const status = document.getElementById('status');
            status.style.display = 'block';
            status.className = 'status ' + type;
            status.textContent = message;
        }
        
        function renderState(state) {
            const cyclesDiv = document.getElementById('cycles');
            cyclesDiv.innerHTML = '';
            
            // Group tasks by cycle
            const tasksByCycle = {};
            state.tasks.forEach(task => {
                const cycle = task.cycle || 1;
                if (!tasksByCycle[cycle]) tasksByCycle[cycle] = [];
                tasksByCycle[cycle].push(task);
            });
            
            Object.keys(tasksByCycle).sort((a, b) => b - a).forEach(cycleNum => {
                const cycleDiv = document.createElement('div');
                cycleDiv.className = 'cycle';
                cycleDiv.innerHTML = '<h3>Cycle ' + cycleNum + '</h3>';
                
                tasksByCycle[cycleNum].forEach(task => {
                    const taskDiv = document.createElement('div');
                    taskDiv.className = 'task ' + task.status;
                    taskDiv.innerHTML = 
                        '<div class="task-info">' +
                            '<div class="task-id">Task ' + task.id + '</div>' +
                            '<div class="task-desc">' + task.description + '</div>' +
                            (task.files.length > 0 ? '<div class="task-files">Files: ' + task.files.join(', ') + '</div>' : '') +
                        '</div>' +
                        '<span class="task-status ' + task.status + '">' + task.status + '</span>';
                    cycleDiv.appendChild(taskDiv);
                });
                
                cyclesDiv.appendChild(cycleDiv);
            });
        }
        
        async function startFurrow() {
            const goal = document.getElementById('goal').value.trim();
            const model = document.getElementById('model').value.trim();
            const resume = document.getElementById('resume').checked;
            
            if (!goal) {
                alert('Please enter a goal');
                return;
            }
            
            isRunning = true;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('cycles').innerHTML = '';
            document.getElementById('log').innerHTML = '';
            
            addLog('Connecting to Furrow...', 'info');
            updateStatus('Starting Furrow...', 'running');
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            
            ws.onopen = () => {
                addLog('Connected. Starting goal: ' + goal, 'success');
                ws.send(JSON.stringify({goal, model: model || null, resume}));
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'state_update') {
                        renderState(data);
                        updateStatus('Cycle ' + data.current_cycle + ' of ' + data.completed_cycles + ' completed', 'running');
                    } else if (data.type === 'log') {
                        addLog(data.message, data.level);
                    } else if (data.type === 'complete') {
                        addLog('Goal complete!', 'success');
                        updateStatus('Goal complete!', 'complete');
                        isRunning = false;
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                    } else if (data.type === 'error') {
                        addLog(data.message, 'error');
                        updateStatus('Error: ' + data.message, 'error');
                        isRunning = false;
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                    }
                } catch (e) {
                    addLog('Raw: ' + event.data, 'info');
                }
            };
            
            ws.onclose = () => {
                addLog('Connection closed', 'warning');
                if (isRunning) {
                    updateStatus('Connection lost', 'error');
                    isRunning = false;
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                }
            };
        }
        
        function stopFurrow() {
            if (ws) {
                ws.close();
            }
            isRunning = false;
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    global active_orchestrator
    await websocket.accept()
    connected_websockets.append(websocket)
    
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        resume = data.get("resume", False)
        
        if not goal:
            await websocket.send_json({"type": "error", "message": "No goal provided"})
            return
        
        if model:
            from furrow.config import settings
            settings.model = model
        
        active_orchestrator = Orchestrator(goal=goal, resume=resume)
        
        # Send initial state
        await broadcast_state_update()
        
        # Run the orchestrator and broadcast updates
        # We need to monkey-patch the orchestrator to send updates
        original_cycle = active_orchestrator._cycle
        
        async def cycle_with_updates():
            await original_cycle()
            await broadcast_state_update()
        
        active_orchestrator._cycle = cycle_with_updates
        
        try:
            await active_orchestrator.run()
            await websocket.send_json({"type": "complete", "message": "Goal complete!"})
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)
        active_orchestrator = None


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
