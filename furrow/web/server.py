from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import CycleState
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Global state for web UI
active_orchestrator: Orchestrator | None = None
websocket_connections: list[WebSocket] = []


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


async def broadcast(message: dict) -> None:
    """Broadcast message to all connected websockets."""
    dead = []
    for ws in websocket_connections:
        try:
            await ws.send_json(message)
        except Exception:  # noqa: BLE001
            dead.append(ws)
    for ws in dead:
        websocket_connections.remove(ws)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow - Autonomous Coding Agent</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #4CAF50; }
        .form-group { margin-bottom: 1rem; }
        input[type="text"] { width: 70%; padding: 0.5rem; font-size: 1rem; background: #16213e; border: 1px solid #0f3460; color: #eee; }
        button { padding: 0.5rem 1.5rem; font-size: 1rem; background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:disabled { background: #666; cursor: not-allowed; }
        #output { background: #16213e; padding: 1rem; border-radius: 4px; font-family: monospace; white-space: pre-wrap; max-height: 500px; overflow-y: auto; border: 1px solid #0f3460; }
        .cycle { border-left: 3px solid #4CAF50; padding-left: 1rem; margin: 1rem 0; }
        .task { margin-left: 2rem; padding: 0.5rem; background: #0f3460; border-radius: 4px; margin: 0.5rem 0; }
        .task.pending { border-left: 3px solid #FFC107; }
        .task.running { border-left: 3px solid #2196F3; }
        .task.completed { border-left: 3px solid #4CAF50; }
        .task.failed { border-left: 3px solid #f44336; }
        .log-entry { margin: 0.25rem 0; padding: 0.25rem; background: #0f3460; border-radius: 2px; font-size: 0.85rem; }
        .log-info { color: #2196F3; }
        .log-success { color: #4CAF50; }
        .log-warning { color: #FFC107; }
        .log-error { color: #f44336; }
        .stats { display: flex; gap: 1rem; margin-bottom: 1rem; }
        .stat-box { background: #16213e; padding: 1rem; border-radius: 4px; border: 1px solid #0f3460; flex: 1; text-align: center; }
        .stat-value { font-size: 2rem; font-weight: bold; color: #4CAF50; }
        .stat-label { font-size: 0.85rem; color: #aaa; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <p>Autonomous coding agent with infinite parallel development loop</p>
    
    <div class="form-group">
        <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
        <button id="startBtn" onclick="startFurrow()">Start</button>
        <button id="stopBtn" onclick="stopFurrow()" disabled>Stop</button>
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <div class="stat-value" id="statCycles">0</div>
            <div class="stat-label">Cycles</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" id="statTasks">0</div>
            <div class="stat-label">Tasks</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" id="statCompleted">0</div>
            <div class="stat-label">Completed</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" id="statFailed">0</div>
            <div class="stat-label">Failed</div>
        </div>
    </div>
    
    <div id="output"></div>
    
    <script>
        let ws = null;
        const output = document.getElementById('output');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        
        function appendLog(msg, type = 'info') {
            const div = document.createElement('div');
            div.className = 'log-entry log-' + type;
            div.textContent = msg;
            output.appendChild(div);
            output.scrollTop = output.scrollHeight;
        }
        
        function updateStats(data) {
            document.getElementById('statCycles').textContent = data.cycles || 0;
            document.getElementById('statTasks').textContent = data.total_tasks || 0;
            document.getElementById('statCompleted').textContent = data.completed || 0;
            document.getElementById('statFailed').textContent = data.failed || 0;
        }
        
        function startFurrow() {
            const goal = document.getElementById('goal').value.trim();
            if (!goal) return;
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            output.textContent = '';
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onopen = () => {
                ws.send(JSON.stringify({goal: goal}));
                appendLog('Connected to Furrow server', 'success');
            };
            ws.onmessage = (ev) => {
                const data = JSON.parse(ev.data);
                handleMessage(data);
            };
            ws.onclose = () => {
                appendLog('Connection closed', 'warning');
                startBtn.disabled = false;
                stopBtn.disabled = true;
            };
            ws.onerror = (err) => {
                appendLog('WebSocket error: ' + err, 'error');
            };
        }
        
        function stopFurrow() {
            if (ws) {
                ws.close();
            }
        }
        
        function handleMessage(data) {
            switch (data.type) {
                case 'cycle_start':
                    appendLog('═══ Cycle ' + data.cycle + ' ═══', 'info');
                    break;
                case 'plan':
                    appendLog('Plan: ' + data.rationale, 'info');
                    data.tasks.forEach(t => {
                        appendLog('  Task ' + t.id + ': ' + t.description, 'info');
                    });
                    updateStats(data.stats);
                    break;
                case 'task_start':
                    appendLog('  → Task ' + data.task_id + ' started', 'info');
                    break;
                case 'task_complete':
                    appendLog('  ✓ Task ' + data.task_id + ' completed', 'success');
                    updateStats(data.stats);
                    break;
                case 'task_failed':
                    appendLog('  ✗ Task ' + data.task_id + ' failed: ' + data.error, 'error');
                    updateStats(data.stats);
                    break;
                case 'test_result':
                    if (data.passed) {
                        appendLog('Tests passed: ' + data.summary, 'success');
                    } else {
                        appendLog('Tests failed: ' + data.summary, 'error');
                        data.failures.forEach(f => appendLog('  - ' + f, 'error'));
                    }
                    break;
                case 'complete':
                    appendLog('Goal complete!', 'success');
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    break;
                case 'error':
                    appendLog('Error: ' + data.message, 'error');
                    break;
                case 'log':
                    appendLog(data.message, data.level || 'info');
                    break;
                default:
                    appendLog(JSON.stringify(data), 'info');
            }
        }
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    global active_orchestrator
    await websocket.accept()
    websocket_connections.append(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        orchestrator = Orchestrator(goal=goal)
        active_orchestrator = orchestrator
        
        # Run orchestrator with websocket updates
        await run_with_websocket(orchestrator, websocket)
        
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:  # noqa: BLE001, S110
            pass
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)
        active_orchestrator = None


async def run_with_websocket(orchestrator: Orchestrator, websocket: WebSocket) -> None:
    """Run orchestrator with websocket updates."""
    await websocket.send_json({
        "type": "cycle_start",
        "cycle": 1,
    })
    
    max_cycles = orchestrator.settings.max_cycles
    while True:
        if max_cycles > 0 and orchestrator.cycles >= max_cycles:
            await websocket.send_json({"type": "log", "message": f"Max cycles ({max_cycles}) reached. Halting.", "level": "warning"})
            break
            
        orchestrator.cycles += 1
        await websocket.send_json({
            "type": "cycle_start",
            "cycle": orchestrator.cycles,
        })
        
        # Planning
        await websocket.send_json({"type": "log", "message": "Planning...", "level": "info"})
        plan = await orchestrator.planner.plan(orchestrator.goal, orchestrator.state)
        
        # Send plan
        tasks_data = [{"id": t.id, "description": t.description, "files": t.files, "status": t.status} for t in plan.tasks]
        await websocket.send_json({
            "type": "plan",
            "rationale": plan.rationale,
            "tasks": tasks_data,
            "stats": {
                "cycles": orchestrator.cycles,
                "total_tasks": len(plan.tasks),
                "completed": 0,
                "failed": 0,
            }
        })
        
        if not plan.tasks:
            await websocket.send_json({"type": "log", "message": "No tasks planned. Goal may be complete.", "level": "warning"})
            cycle_state = CycleState(cycle=orchestrator.cycles, goal=orchestrator.goal, plan=plan)
            orchestrator.state.add_cycle(cycle_state)
            orchestrator.state.save(orchestrator.settings.state_file)
            continue
        
        # Execute tasks
        pending_tasks = [t for t in plan.tasks if t.status == "pending"]
        if not pending_tasks:
            await websocket.send_json({"type": "log", "message": "All tasks already completed.", "level": "warning"})
            cycle_state = CycleState(cycle=orchestrator.cycles, goal=orchestrator.goal, plan=plan)
            orchestrator.state.add_cycle(cycle_state)
            orchestrator.state.save(orchestrator.settings.state_file)
            continue
        
        completed_count = 0
        failed_count = 0
        
        for task in pending_tasks:
            await websocket.send_json({"type": "task_start", "task_id": task.id})
            try:
                worker = WorkerAgent(task=task, client=orchestrator.client, settings=orchestrator.settings)
                result = await worker.run()
                task.status = "completed"
                task.result = result
                completed_count += 1
                await websocket.send_json({
                    "type": "task_complete",
                    "task_id": task.id,
                    "stats": {
                        "cycles": orchestrator.cycles,
                        "total_tasks": len(plan.tasks),
                        "completed": completed_count,
                        "failed": failed_count,
                    }
                })
            except Exception as e:  # noqa: BLE001
                task.status = "failed"
                task.result = str(e)
                failed_count += 1
                await websocket.send_json({
                    "type": "task_failed",
                    "task_id": task.id,
                    "error": str(e),
                    "stats": {
                        "cycles": orchestrator.cycles,
                        "total_tasks": len(plan.tasks),
                        "completed": completed_count,
                        "failed": failed_count,
                    }
                })
        
        # Testing
        await websocket.send_json({"type": "log", "message": "Testing...", "level": "info"})
        tester = TesterAgent(client=orchestrator.client, settings=orchestrator.settings)
        test_result = await tester.run(orchestrator.goal, plan.tasks)
        
        cycle_state = CycleState(
            cycle=orchestrator.cycles,
            goal=orchestrator.goal,
            plan=plan,
            test_result=test_result,
        )
        orchestrator.state.add_cycle(cycle_state)
        orchestrator.state.save(orchestrator.settings.state_file)
        
        await websocket.send_json({
            "type": "test_result",
            "passed": test_result.passed,
            "summary": test_result.summary,
            "failures": test_result.failures,
        })
        
        if test_result.passed:
            await websocket.send_json({"type": "log", "message": f"Tests passed: {test_result.summary}", "level": "success"})
        else:
            await websocket.send_json({"type": "log", "message": f"Tests failed: {test_result.summary}", "level": "error"})
            for failure in test_result.failures:
                await websocket.send_json({"type": "log", "message": f"  • {failure}", "level": "error"})
            await websocket.send_json({"type": "log", "message": "Will attempt fix in next cycle.", "level": "warning"})
            orchestrator.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
        
        if orchestrator._is_done():
            await websocket.send_json({"type": "complete", "message": "Goal complete!"})
            break
    
    await websocket.send_json({"type": "log", "message": "Furrow finished.", "level": "success"})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
