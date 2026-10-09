from __future__ import annotations

import asyncio
import json
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


# Global orchestrator reference for WebSocket updates
current_orchestrator: Orchestrator | None = None
connected_websockets: set[WebSocket] = set()


async def broadcast(message: dict) -> None:
    """Broadcast message to all connected WebSocket clients."""
    disconnected = set()
    for ws in connected_websockets:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.add(ws)
    connected_websockets -= disconnected


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup on shutdown
    for ws in connected_websockets:
        try:
            await ws.close()
        except Exception:
            pass


app = FastAPI(title="Furrow", lifespan=lifespan)


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
        .panel { background: #16213e; border-radius: 8px; padding: 20px; margin-bottom: 20px; border: 1px solid #0f3460; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 10px; border: 1px solid #0f3460; border-radius: 4px; background: #0f0f23; color: #eee; font-size: 14px; }
        button { background: #4CAF50; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 16px; font-weight: bold; }
        button:hover { background: #45a049; }
        button:disabled { background: #666; cursor: not-allowed; }
        #status { padding: 15px; border-radius: 4px; margin-bottom: 20px; }
        .status-running { background: #0f3460; border: 1px solid #4CAF50; }
        .status-complete { background: #1b5e20; border: 1px solid #4CAF50; }
        .status-error { background: #b71c1c; border: 1px solid #f44336; }
        .cycles { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; }
        .cycle-card { background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 15px; }
        .cycle-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .cycle-number { font-size: 1.2em; font-weight: bold; color: #4CAF50; }
        .cycle-status { padding: 4px 8px; border-radius: 4px; font-size: 0.85em; }
        .status-planning { background: #ff9800; color: #000; }
        .status-executing { background: #2196F3; color: #fff; }
        .status-testing { background: #9C27B0; color: #fff; }
        .status-complete { background: #4CAF50; color: #fff; }
        .status-failed { background: #f44336; color: #fff; }
        .task-list { list-style: none; padding: 0; margin: 0; }
        .task-item { padding: 8px; margin: 5px 0; background: #0f0f23; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; }
        .task-desc { flex: 1; margin: 0 10px; }
        .task-status { font-size: 0.85em; padding: 2px 6px; border-radius: 3px; }
        .task-completed { background: #4CAF50; }
        .task-failed { background: #f44336; }
        .task-pending { background: #ff9800; color: #000; }
        .usage { background: #0f0f23; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.9em; }
        .log { background: #0f0f23; border: 1px solid #0f3460; border-radius: 4px; padding: 15px; max-height: 400px; overflow-y: auto; font-family: monospace; font-size: 0.85em; line-height: 1.5; }
        .log-entry { margin: 5px 0; }
        .log-info { color: #2196F3; }
        .log-success { color: #4CAF50; }
        .log-warning { color: #ff9800; }
        .log-error { color: #f44336; }
        .goal-display { background: #0f3460; padding: 15px; border-radius: 4px; margin-bottom: 20px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>🌾 Furrow</h1>
    <p>Autonomous coding agent with infinite parallel development loop</p>
    
    <div class="panel">
        <h2>Start New Session</h2>
        <form id="startForm">
            <div class="form-group">
                <label for="goal">Goal</label>
                <textarea id="goal" rows="3" placeholder="Describe what you want to build..." required></textarea>
            </div>
            <div class="form-group">
                <label for="model">Model (optional)</label>
                <input id="model" type="text" placeholder="e.g., claude-sonnet-4-20250514" />
            </div>
            <button type="submit" id="startBtn">Start Furrow</button>
        </form>
    </div>
    
    <div id="status" class="status-running" style="display:none;">
        <strong>Status:</strong> <span id="statusText">Initializing...</span>
    </div>
    
    <div id="goalDisplay" class="goal-display" style="display:none;"></div>
    
    <div class="panel">
        <h2>Cycles</h2>
        <div id="cycles" class="cycles"></div>
    </div>
    
    <div class="panel">
        <h2>Token Usage</h2>
        <div id="usage" class="usage">Not started</div>
    </div>
    
    <div class="panel">
        <h2>Live Log</h2>
        <div id="log" class="log"></div>
    </div>

    <script>
        let ws = null;
        let currentCycle = 0;
        
        const form = document.getElementById('form');
        const startForm = document.getElementById('startForm');
        const startBtn = document.getElementById('startBtn');
        const statusDiv = document.getElementById('status');
        const statusText = document.getElementById('statusText');
        const goalDisplay = document.getElementById('goalDisplay');
        const cyclesDiv = document.getElementById('cycles');
        const usageDiv = document.getElementById('usage');
        const logDiv = document.getElementById('log');
        
        function addLog(message, type = 'info') {
            const entry = document.createElement('div');
            entry.className = 'log-entry log-' + type;
            entry.textContent = '[' + new Date().toLocaleTimeString() + '] ' + message;
            logDiv.appendChild(entry);
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        function updateStatus(text, className = 'status-running') {
            statusDiv.style.display = 'block';
            statusDiv.className = className;
            statusText.textContent = text;
        }
        
        function renderCycle(cycleData) {
            let card = document.getElementById('cycle-' + cycleData.number);
            if (!card) {
                card = document.createElement('div');
                card.id = 'cycle-' + cycleData.number;
                card.className = 'cycle-card';
                cyclesDiv.insertBefore(card, cyclesDiv.firstChild);
            }
            
            const tasksHtml = cycleData.tasks.map(t => 
                '<li class="task-item">' +
                '<span class="task-status task-' + t.status + '">' + t.status.toUpperCase() + '</span>' +
                '<span class="task-desc">' + t.description + '</span>' +
                '</li>'
            ).join('');
            
            card.innerHTML = 
                '<div class="cycle-header">' +
                '<span class="cycle-number">Cycle ' + cycleData.number + '</span>' +
                '<span class="cycle-status status-' + cycleData.status + '">' + cycleData.status.toUpperCase() + '</span>' +
                '</div>' +
                '<ul class="task-list">' + tasksHtml + '</ul>' +
                (cycleData.testResult ? 
                    '<div style="margin-top:10px;padding:10px;background:#0f0f23;border-radius:4px;">' +
                    '<strong>Test Result:</strong> ' + (cycleData.testResult.passed ? '✅ PASSED' : '❌ FAILED') + ' - ' + cycleData.testResult.summary +
                    '</div>' : '');
        }
        
        function updateUsage(usage) {
            usageDiv.textContent = usage;
        }
        
        startForm.onsubmit = async (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value.trim();
            const model = document.getElementById('model').value.trim();
            if (!goal) return;
            
            startBtn.disabled = true;
            startBtn.textContent = 'Starting...';
            cyclesDiv.innerHTML = '';
            logDiv.innerHTML = '';
            currentCycle = 0;
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onopen = () => {
                addLog('Connected to Furrow', 'success');
                ws.send(JSON.stringify({goal, model: model || null}));
            };
            ws.onmessage = (ev) => {
                try {
                    const msg = JSON.parse(ev.data);
                    handleMessage(msg);
                } catch (err) {
                    addLog('Error parsing message: ' + err, 'error');
                }
            };
            ws.onclose = () => {
                addLog('Connection closed', 'warning');
                startBtn.disabled = false;
                startBtn.textContent = 'Start Furrow';
            };
            ws.onerror = (err) => {
                addLog('WebSocket error', 'error');
            };
        };
        
        function handleMessage(msg) {
            switch (msg.type) {
                case 'cycle_start':
                    currentCycle = msg.cycle;
                    updateStatus('Cycle ' + msg.cycle + ': Planning...', 'status-running');
                    addLog('Starting cycle ' + msg.cycle, 'info');
                    goalDisplay.style.display = 'block';
                    goalDisplay.textContent = 'Goal: ' + msg.goal;
                    break;
                case 'plan':
                    updateStatus('Cycle ' + currentCycle + ': Executing ' + msg.tasks.length + ' tasks...', 'status-running');
                    addLog('Plan created with ' + msg.tasks.length + ' tasks', 'info');
                    msg.tasks.forEach(t => addLog('  - ' + t.id + ': ' + t.description, 'info'));
                    renderCycle({number: currentCycle, status: 'executing', tasks: msg.tasks.map(t => ({...t, status: 'pending'}))});
                    break;
                case 'task_complete':
                    addLog('Task ' + msg.task.id + ' completed', 'success');
                    renderCycle({number: currentCycle, status: 'executing', tasks: msg.tasks});
                    break;
                case 'task_failed':
                    addLog('Task ' + msg.task.id + ' failed: ' + msg.error, 'error');
                    renderCycle({number: currentCycle, status: 'executing', tasks: msg.tasks});
                    break;
                case 'testing':
                    updateStatus('Cycle ' + currentCycle + ': Running tests...', 'status-running');
                    addLog('Running tests...', 'info');
                    renderCycle({number: currentCycle, status: 'testing', tasks: msg.tasks});
                    break;
                case 'test_result':
                    const passed = msg.result.passed;
                    updateStatus('Cycle ' + currentCycle + ': ' + (passed ? 'Tests Passed ✓' : 'Tests Failed ✗'), passed ? 'status-complete' : 'status-error');
                    addLog('Tests ' + (passed ? 'PASSED' : 'FAILED') + ': ' + msg.result.summary, passed ? 'success' : 'error');
                    if (msg.result.failures.length) {
                        msg.result.failures.forEach(f => addLog('  Failure: ' + f, 'error'));
                    }
                    renderCycle({number: currentCycle, status: passed ? 'complete' : 'failed', tasks: msg.tasks, testResult: msg.result});
                    break;
                case 'cycle_complete':
                    addLog('Cycle ' + currentCycle + ' complete', 'success');
                    break;
                case 'done':
                    updateStatus('Goal Complete! 🎉', 'status-complete');
                    addLog('Goal completed successfully!', 'success');
                    break;
                case 'max_cycles':
                    updateStatus('Max cycles reached', 'status-warning');
                    addLog('Max cycles reached, stopping', 'warning');
                    break;
                case 'usage':
                    updateUsage(msg.usage);
                    break;
                case 'error':
                    updateStatus('Error: ' + msg.message, 'status-error');
                    addLog('Error: ' + msg.message, 'error');
                    break;
                case 'log':
                    addLog(msg.message, msg.level || 'info');
                    break;
            }
        }
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    connected_websockets.add(websocket)
    
    global current_orchestrator
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        
        if model:
            settings.model = model
        
        current_orchestrator = Orchestrator(goal=goal)
        
        # Patch the orchestrator to send WebSocket updates
        original_cycle = current_orchestrator._cycle
        
        async def cycle_with_updates():
            await broadcast({"type": "cycle_start", "cycle": current_orchestrator.cycles + 1, "goal": current_orchestrator.goal})
            
            # Planning
            context = current_orchestrator._build_context()
            plan = await current_orchestrator.planner.plan(current_orchestrator.goal, context)
            await broadcast({"type": "plan", "tasks": [t.model_dump() for t in plan.tasks]})
            
            if not plan.tasks:
                await broadcast({"type": "log", "message": "No tasks planned. Goal may be complete.", "level": "warning"})
                return
            
            # Filter pending tasks
            pending_tasks = [t for t in plan.tasks if t.id not in {h.id for h in current_orchestrator.task_history if h.status == "completed"}]
            
            if not pending_tasks:
                await broadcast({"type": "log", "message": "All planned tasks already completed.", "level": "success"})
                return
            
            # Execute tasks
            async def run_task(task):
                try:
                    result = await WorkerAgent(task=task, client=current_orchestrator.client, settings=current_orchestrator.settings).run()
                    task.status = "completed"
                    task.result = result
                    await broadcast({"type": "task_complete", "task": task.model_dump(), "tasks": [t.model_dump() for t in pending_tasks]})
                except Exception as e:
                    task.status = "failed"
                    task.result = str(e)
                    await broadcast({"type": "task_failed", "task": task.model_dump(), "error": str(e), "tasks": [t.model_dump() for t in pending_tasks]})
                finally:
                    current_orchestrator._update_task_history(task)
            
            await asyncio.gather(*[run_task(task) for task in pending_tasks])
            
            # Testing
            await broadcast({"type": "testing", "tasks": [t.model_dump() for t in pending_tasks]})
            test_result = await TesterAgent(client=current_orchestrator.client, settings=current_orchestrator.settings).run(current_orchestrator.goal, pending_tasks)
            
            await broadcast({"type": "test_result", "result": test_result.model_dump(), "tasks": [t.model_dump() for t in pending_tasks]})
            
            if test_result.passed:
                await broadcast({"type": "log", "message": f"Tests passed: {test_result.summary}", "level": "success"})
            else:
                await broadcast({"type": "log", "message": f"Tests failed: {test_result.summary}", "level": "error"})
                for failure in test_result.failures:
                    await broadcast({"type": "log", "message": f"  Failure: {failure}", "level": "error"})
                current_orchestrator.goal = f"{current_orchestrator.goal}\n\nPrevious test failures to fix:\n" + "\n".join(f"- {f}" for f in test_result.failures)
            
            await broadcast({"type": "cycle_complete", "cycle": current_orchestrator.cycles})
            
            # Send token usage
            await broadcast({"type": "usage", "usage": current_orchestrator.client.get_usage_summary()})
        
        # Run the loop
        while True:
            current_orchestrator.cycles += 1
            
            if settings.max_cycles > 0 and current_orchestrator.cycles > settings.max_cycles:
                await broadcast({"type": "max_cycles", "message": f"Max cycles ({settings.max_cycles}) reached"})
                break
            
            await cycle_with_updates()
            current_orchestrator._save_state()
            
            # Auto-commit
            from furrow.core.orchestrator import _is_git_repo, _git_commit
            if _is_git_repo(settings.workspace):
                commit_msg = f"furrow: cycle {current_orchestrator.cycles} - {current_orchestrator._summarize_cycle()}"
                if _git_commit(settings.workspace, commit_msg):
                    await broadcast({"type": "log", "message": f"Committed: {commit_msg}", "level": "info"})
            
            if current_orchestrator._is_done():
                await broadcast({"type": "done", "message": "Goal complete"})
                break
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await broadcast({"type": "error", "message": str(e)})
    finally:
        connected_websockets.discard(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
