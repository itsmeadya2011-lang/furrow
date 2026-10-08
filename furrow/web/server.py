from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from furrow.config import StateManager, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")

# Store active websocket connections
active_connections: list[WebSocket] = []


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    provider: str | None = None
    max_cycles: int | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
        h1 { color: #4CAF50; }
        .form-group { margin-bottom: 15px; }
        input[type="text"] { width: 70%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        #output { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; max-height: 500px; overflow-y: auto; }
        .cycle { border-left: 3px solid #2196F3; padding-left: 10px; margin: 10px 0; }
        .task-completed { color: #4CAF50; }
        .task-failed { color: #f44336; }
        .task-running { color: #ff9800; }
        .test-passed { color: #4CAF50; font-weight: bold; }
        .test-failed { color: #f44336; font-weight: bold; }
        .status-bar { display: flex; gap: 20px; margin-bottom: 20px; padding: 10px; background: #f5f5f5; border-radius: 5px; }
        .status-item { font-weight: bold; }
    </style>
</head>
<body>
    <h1>Furrow - Autonomous Coding Agent</h1>
    
    <div class="status-bar">
        <div class="status-item">Cycles: <span id="cycle-count">0</span></div>
        <div class="status-item">Tasks: <span id="task-count">0</span> completed</div>
        <div class="status-item">Tests: <span id="test-status">pending</span></div>
    </div>
    
    <form id="form">
        <div class="form-group">
            <input id="goal" placeholder="Enter your goal (e.g., 'Add JWT authentication to the API')" required />
        </div>
        <div class="form-group">
            <button type="submit" id="start-btn">Start Furrow</button>
            <button type="button" id="stop-btn" disabled>Stop</button>
        </div>
    </form>
    
    <div id="output"></div>
    
    <script>
        const form = document.getElementById('form');
        const output = document.getElementById('output');
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        const cycleCount = document.getElementById('cycle-count');
        const taskCount = document.getElementById('task-count');
        const testStatus = document.getElementById('test-status');
        
        let ws = null;
        let isRunning = false;
        
        function appendOutput(text, className = '') {
            const div = document.createElement('div');
            div.textContent = text;
            if (className) div.className = className;
            output.appendChild(div);
            output.scrollTop = output.scrollHeight;
        }
        
        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value;
            if (!goal) return;
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            isRunning = true;
            output.textContent = '';
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                try {
                    const data = JSON.parse(ev.data);
                    handleMessage(data);
                } catch {
                    appendOutput(ev.data);
                }
            };
            ws.onclose = () => {
                appendOutput('\\n[Connection closed]');
                startBtn.disabled = false;
                stopBtn.disabled = true;
                isRunning = false;
            };
            ws.onerror = (err) => {
                appendOutput('[WebSocket error]');
            };
            
            ws.send(JSON.stringify({goal: goal}));
        };
        
        stopBtn.onclick = () => {
            if (ws) {
                ws.send(JSON.stringify({type: 'stop'}));
                ws.close();
            }
        };
        
        function handleMessage(data) {
            if (data.type === 'cycle_start') {
                cycleCount.textContent = data.cycle;
                appendOutput(`\\n═══ Cycle ${data.cycle} ═══`, 'cycle');
            } else if (data.type === 'plan') {
                appendOutput(`\\nPlan: ${data.rationale}`);
                data.tasks.forEach(t => {
                    appendOutput(`  • ${t.id}: ${t.description}`, 'task-running');
                });
            } else if (data.type === 'task_complete') {
                const cls = data.status === 'completed' ? 'task-completed' : 'task-failed';
                appendOutput(`  ${data.status.toUpperCase()}: ${data.task_id} - ${data.result}`, cls);
                updateTaskCount();
            } else if (data.type === 'test_result') {
                const cls = data.passed ? 'test-passed' : 'test-failed';
                appendOutput(`\\nTests: ${data.passed ? 'PASSED' : 'FAILED'} - ${data.summary}`, cls);
                testStatus.textContent = data.passed ? 'passed' : 'failed';
                if (!data.passed && data.failures) {
                    data.failures.forEach(f => appendOutput(`  ✗ ${f}`, 'task-failed'));
                }
            } else if (data.type === 'summary') {
                appendOutput(`\\n${data.message}`);
            } else if (data.type === 'done') {
                appendOutput(`\\n${data.message}`);
                startBtn.disabled = false;
                stopBtn.disabled = true;
                isRunning = false;
            } else if (data.type === 'error') {
                appendOutput(`\\n[ERROR] ${data.message}`, 'task-failed');
            } else if (data.type === 'log') {
                appendOutput(data.message);
            }
        }
        
        function updateTaskCount() {
            const completed = document.querySelectorAll('.task-completed').length;
            taskCount.textContent = completed;
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
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        provider = data.get("provider")
        max_cycles = data.get("max_cycles")
        
        if model:
            settings.model = model
        if provider:
            settings.provider = provider
        if max_cycles is not None:
            settings.max_cycles = max_cycles
        
        client = LLMClient()
        orchestrator = Orchestrator(goal=goal, client=client, settings=settings)
        
        # Run with custom output
        await orchestrator.run_with_callback(lambda msg: websocket.send_json(msg))
        
        await websocket.send_json({"type": "done", "message": "Goal complete. Furrow stopped."})
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("Error in websocket endpoint")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except (WebSocketDisconnect, RuntimeError):
            pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.get("/api/status")
async def get_status() -> dict:
    state = StateManager(settings.state_file)
    return {
        "goal": state.current_goal,
        "cycles": len(state.cycles),
        "tasks": [
            {
                "id": t.id,
                "description": t.description,
                "status": t.status,
                "cycle": t.cycle,
                "attempts": t.attempts,
            }
            for t in state.all_tasks
        ],
        "history": [
            {
                "cycle": c.cycle,
                "goal": c.goal,
                "test_passed": c.test_result.passed if c.test_result else None,
                "timestamp": c.timestamp,
            }
            for c in state.cycles
        ],
    }


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
