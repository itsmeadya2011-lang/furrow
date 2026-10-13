from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import State, get_logger, settings, setup_logging
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

setup_logging()

logger = get_logger("server")

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    max_cycles: int | None = None


class TaskStatusModel(BaseModel):
    id: str
    description: str
    files: list[str] = []
    dependencies: list[str] = []
    status: str
    result: str | None = None
    cycle: int = 0


class StateResponse(BaseModel):
    goal: str
    cycles: int
    current_cycle: int
    tasks: list[TaskStatusModel]
    is_complete: bool


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        .form-group { margin-bottom: 1rem; }
        label { display: block; margin-bottom: 0.5rem; font-weight: bold; }
        input, select { width: 100%; max-width: 500px; padding: 0.5rem; font-size: 1rem; }
        button { padding: 0.75rem 1.5rem; font-size: 1rem; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #45a049; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .status { padding: 1rem; margin: 1rem 0; border-radius: 4px; }
        .status.running { background: #e3f2fd; border: 1px solid #2196F3; }
        .status.complete { background: #e8f5e9; border: 1px solid #4CAF50; }
        .status.error { background: #ffebee; border: 1px solid #f44336; }
        .task { padding: 0.75rem; margin: 0.5rem 0; border-radius: 4px; border-left: 4px solid; }
        .task.pending { background: #fff3e0; border-color: #ff9800; }
        .task.in_progress { background: #e3f2fd; border-color: #2196F3; }
        .task.completed { background: #e8f5e9; border-color: #4CAF50; }
        .task.failed { background: #ffebee; border-color: #f44336; }
        .task-header { display: flex; justify-content: space-between; margin-bottom: 0.5rem; }
        .task-id { font-weight: bold; color: #666; }
        .task-cycle { color: #999; font-size: 0.85rem; }
        .log { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; font-family: monospace; max-height: 400px; overflow-y: auto; white-space: pre-wrap; }
        .log-line { margin: 0.25rem 0; }
        .log-info { color: #4fc3f7; }
        .log-success { color: #81c784; }
        .log-warning { color: #ffb74d; }
        .log-error { color: #e57373; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    
    <div id="status" class="status" style="display:none;"></div>
    
    <form id="form" class="form-group">
        <label for="goal">Goal</label>
        <input id="goal" placeholder="Enter goal (e.g., Add JWT auth to the API)" required />
        
        <div class="form-group">
            <label for="model">Model (optional)</label>
            <input id="model" placeholder="e.g., claude-3-5-sonnet-20241022" />
        </div>
        
        <div class="form-group">
            <label for="max_cycles">Max Cycles (0 = unlimited)</label>
            <input id="max_cycles" type="number" value="0" min="0" />
        </div>
        
        <button type="submit" id="startBtn">Start</button>
        <button type="button" id="stopBtn" disabled>Stop</button>
    </form>
    
    <h2>Tasks</h2>
    <div id="tasks"></div>
    
    <h2>Log</h2>
    <div id="log" class="log"></div>
    
    <script>
        const form = document.getElementById('form');
        const statusDiv = document.getElementById('status');
        const tasksDiv = document.getElementById('tasks');
        const logDiv = document.getElementById('log');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        
        let ws = null;
        let isRunning = false;
        
        function addLog(message, className = '') {
            const line = document.createElement('div');
            line.className = 'log-line ' + className;
            line.textContent = message;
            logDiv.appendChild(line);
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        function updateStatus(message, className = '') {
            statusDiv.textContent = message;
            statusDiv.className = 'status ' + className;
            statusDiv.style.display = 'block';
        }
        
        function renderTasks(tasks) {
            tasksDiv.innerHTML = '';
            tasks.forEach(task => {
                const div = document.createElement('div');
                div.className = 'task ' + task.status;
                div.innerHTML = `
                    <div class="task-header">
                        <span class="task-id">${task.id}</span>
                        <span class="task-cycle">Cycle ${task.cycle}</span>
                    </div>
                    <div>${task.description}</div>
                    ${task.files.length ? '<div style="color:#666;font-size:0.85rem;">Files: ' + task.files.join(', ') + '</div>' : ''}
                    ${task.dependencies.length ? '<div style="color:#666;font-size:0.85rem;">Deps: ' + task.dependencies.join(', ') + '</div>' : ''}
                    ${task.result ? '<div style="margin-top:0.5rem;color:#666;">' + task.result + '</div>' : ''}
                `;
                tasksDiv.appendChild(div);
            });
        }
        
        async function fetchState() {
            try {
                const resp = await fetch('/api/state');
                const data = await resp.json();
                renderTasks(data.tasks);
                if (data.is_complete) {
                    updateStatus('Goal complete!', 'complete');
                    isRunning = false;
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                }
            } catch (e) {
                console.error('Failed to fetch state', e);
            }
        }
        
        form.onsubmit = async (e) => {
            e.preventDefault();
            if (isRunning) return;
            
            const goal = document.getElementById('goal').value;
            const model = document.getElementById('model').value || null;
            const max_cycles = parseInt(document.getElementById('max_cycles').value) || null;
            
            isRunning = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            logDiv.innerHTML = '';
            updateStatus('Starting...', 'running');
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onopen = () => {
                ws.send(JSON.stringify({goal, model, max_cycles}));
            };
            ws.onmessage = (ev) => {
                try {
                    const data = JSON.parse(ev.data);
                    if (data.type === 'log') {
                        addLog(data.message, data.level);
                    } else if (data.type === 'state') {
                        renderTasks(data.tasks);
                        if (data.is_complete) {
                            updateStatus('Goal complete!', 'complete');
                            isRunning = false;
                            startBtn.disabled = false;
                            stopBtn.disabled = true;
                        } else {
                            updateStatus(`Cycle ${data.current_cycle} running...`, 'running');
                        }
                    } else if (data.type === 'error') {
                        addLog(data.message, 'log-error');
                        updateStatus('Error: ' + data.message, 'error');
                        isRunning = false;
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                    }
                } catch (e) {
                    addLog(ev.data);
                }
            };
            ws.onclose = () => {
                if (isRunning) {
                    addLog('Connection closed unexpectedly', 'log-warning');
                }
                isRunning = false;
                startBtn.disabled = false;
                stopBtn.disabled = true;
            };
        };
        
        stopBtn.onclick = () => {
            if (ws) {
                ws.close();
            }
            isRunning = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            updateStatus('Stopped by user', 'warning');
        };
        
        // Initial state fetch
        fetchState();
        setInterval(fetchState, 5000);
    </script>
</body>
</html>
""")


@app.get("/api/state")
async def get_state() -> StateResponse:
    state = State.from_file(settings.state_file)
    if not state:
        return StateResponse(goal="", cycles=0, current_cycle=0, tasks=[], is_complete=False)
    
    tasks = [
        TaskStatusModel(
            id=t.id,
            description=t.description,
            files=t.files,
            dependencies=t.dependencies,
            status=t.status.value,
            result=t.result,
            cycle=t.cycle,
        )
        for t in state.tasks
    ]
    
    return StateResponse(
        goal=state.goal,
        cycles=state.cycles,
        current_cycle=state.current_cycle,
        tasks=tasks,
        is_complete=state.is_goal_complete(),
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    client = LLMClient()
    
    async def send_log(message: str, level: str = "info") -> None:
        await websocket.send_json({"type": "log", "message": message, "level": level})
    
    async def send_state(state: State) -> None:
        tasks = [
            {
                "id": t.id,
                "description": t.description,
                "files": t.files,
                "dependencies": t.dependencies,
                "status": t.status.value,
                "result": t.result,
                "cycle": t.cycle,
            }
            for t in state.tasks
        ]
        await websocket.send_json({
            "type": "state",
            "current_cycle": state.current_cycle,
            "tasks": tasks,
            "is_complete": state.is_goal_complete(),
        })
    
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        max_cycles = data.get("max_cycles")
        
        if model:
            settings.model = model
        if max_cycles:
            settings.max_cycles = max_cycles
        
        await send_log(f"Starting Furrow with goal: {goal}")
        
        orchestrator = Orchestrator(goal=goal, client=client)
        
        max_cycles_limit = settings.max_cycles if settings.max_cycles > 0 else float('inf')
        
        while orchestrator.state.current_cycle < max_cycles_limit:
            orchestrator.state.current_cycle += 1
            await send_log(f"═══ Cycle {orchestrator.state.current_cycle} ═══", "info")
            await send_state(orchestrator.state)
            
            await orchestrator._cycle()
            await send_state(orchestrator.state)
            
            if orchestrator.state.is_goal_complete():
                await send_log("Goal complete. Halting.", "success")
                break
            
            if orchestrator.state.current_cycle >= max_cycles_limit:
                await send_log(f"Max cycles ({max_cycles_limit}) reached.", "warning")
                break
        
    except WebSocketDisconnect:
        pass
    # Report any unexpected error that occurred during the session.
    except Exception as e:  # noqa: BLE001
        await send_log(f"Error: {e}", "error")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:  # noqa: BLE001
            logger.debug("websocket_send_error_failed", error=str(e))


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
