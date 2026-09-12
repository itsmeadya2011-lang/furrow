from __future__ import annotations

import asyncio
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, TaskModel, Plan, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent

app = FastAPI(title="Furrow")

# Store active orchestrator for status updates
active_orchestrator: Orchestrator | None = None


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


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
        .status { padding: 10px; border-radius: 4px; margin: 10px 0; }
        .status.running { background: #e3f2fd; color: #1565c0; }
        .status.completed { background: #e8f5e9; color: #2e7d32; }
        .status.failed { background: #ffebee; color: #c62828; }
        .status.pending { background: #fff3e0; color: #ef6c00; }
        .cycle { border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin: 10px 0; }
        .cycle-header { font-weight: bold; color: #1976d2; margin-bottom: 10px; }
        .task { padding: 8px; margin: 5px 0; border-radius: 4px; }
        .task.completed { background: #e8f5e9; }
        .task.failed { background: #ffebee; }
        .task.pending { background: #fff3e0; }
        .task.running { background: #e3f2fd; }
        .log { background: #f5f5f5; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 12px; max-height: 400px; overflow-y: auto; }
        form { margin: 20px 0; }
        input[type="text"] { padding: 10px; width: 60%; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; background: #1976d2; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #1565c0; }
        button:disabled { background: #90caf9; cursor: not-allowed; }
        .summary { display: flex; gap: 20px; margin: 20px 0; }
        .stat { padding: 15px; background: #f5f5f5; border-radius: 4px; flex: 1; text-align: center; }
        .stat-value { font-size: 32px; font-weight: bold; color: #1976d2; }
        .stat-label { font-size: 14px; color: #666; }
    </style>
</head>
<body>
    <h1>Furrow - Autonomous Coding Agent</h1>
    
    <form id="form">
        <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
        <button type="submit" id="startBtn">Start</button>
    </form>
    
    <div id="status" class="status pending">Waiting to start...</div>
    
    <div class="summary">
        <div class="stat">
            <div class="stat-value" id="cycles">0</div>
            <div class="stat-label">Cycles</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="completed">0</div>
            <div class="stat-label">Completed</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="failed">0</div>
            <div class="stat-label">Failed</div>
        </div>
        <div class="stat">
            <div class "stat-value" id="pending">0</div>
            <div class="stat-label">Pending</div>
        </div>
    </div>
    
    <div id="cycles"></div>
    
    <div class="log" id="log"></div>
    
    <script>
        const form = document.getElementById('form');
        const statusEl = document.getElementById('status');
        const logEl = document.getElementById('log');
        const cyclesEl = document.getElementById('cycles');
        const startBtn = document.getElementById('startBtn');
        
        let ws = null;
        
        function updateStatus(text, className) {
            statusEl.textContent = text;
            statusEl.className = 'status ' + className;
        }
        
        function addLog(text) {
            const time = new Date().toLocaleTimeString();
            logEl.textContent += '[' + time + '] ' + text + '\\n';
            logEl.scrollTop = logEl.scrollHeight;
        }
        
        function updateStats(data) {
            document.getElementById('cycles').textContent = data.cycles || 0;
            document.getElementById('completed').textContent = data.completed || 0;
            document.getElementById('failed').textContent = data.failed || 0;
            document.getElementById('pending').textContent = data.pending || 0;
        }
        
        function renderCycle(cycle) {
            const div = document.createElement('div');
            div.className = 'cycle';
            
            let html = '<div class="cycle-header">Cycle ' + cycle.number + ': ' + cycle.goal + '</div>';
            
            if (cycle.plan) {
                html += '<div><strong>Plan:</strong> ' + cycle.plan.rationale + '</div>';
                html += '<div><strong>Tasks:</strong></div>';
                for (const task of cycle.plan.tasks) {
                    const taskClass = task.status || 'pending';
                    html += '<div class="task ' + taskClass + '">';
                    html += '  <strong>' + task.id + ':</strong> ' + task.description;
                    if (task.files.length > 0) {
                        html += ' <em>(files: ' + task.files.join(', ') + ')</em>';
                    }
                    if (task.result) {
                        html += ' <br><small>Result: ' + task.result.substring(0, 200) + '</small>';
                    }
                    html += '</div>';
                }
            }
            
            if (cycle.test_result) {
                const tr = cycle.test_result;
                const testClass = tr.passed ? 'completed' : 'failed';
                html += '<div class="task ' + testClass + '">';
                html += '  <strong>Test Result:</strong> ' + tr.summary;
                if (tr.failures.length > 0) {
                    html += ' <br><small>Failures: ' + tr.failures.join('; ') + '</small>';
                }
                html += '</div>';
            }
            
            div.innerHTML = html;
            cyclesEl.insertBefore(div, cyclesEl.firstChild);
        }
        
        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value;
            
            startBtn.disabled = true;
            updateStatus('Connecting...', 'running');
            logEl.textContent = '';
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onopen = () => {
                updateStatus('Running...', 'running');
                ws.send(JSON.stringify({goal: goal}));
            };
            ws.onmessage = (ev) => {
                try {
                    const data = JSON.parse(ev.data);
                    handleMessage(data);
                } catch (err) {
                    addLog('Error parsing message: ' + err);
                }
            };
            ws.onclose = () => {
                updateStatus('Disconnected', 'pending');
                startBtn.disabled = false;
            };
            ws.onerror = (err) => {
                addLog('WebSocket error: ' + err);
            };
        };
        
        function handleMessage(data) {
            switch (data.type) {
                case 'cycle_start':
                    addLog('Starting cycle ' + data.cycle);
                    break;
                case 'plan':
                    addLog('Plan created with ' + data.plan.tasks.length + ' tasks');
                    break;
                case 'task_start':
                    addLog('Executing task ' + data.task.id + ': ' + data.task.description);
                    break;
                case 'task_complete':
                    addLog('Task ' + data.task.id + ' completed');
                    break;
                case 'task_failed':
                    addLog('Task ' + data.task.id + ' failed: ' + data.error);
                    break;
                case 'test_result':
                    addLog('Tests ' + (data.result.passed ? 'passed' : 'failed') + ': ' + data.result.summary);
                    break;
                case 'cycle_complete':
                    addLog('Cycle ' + data.cycle + ' complete');
                    renderCycle(data.cycle_data);
                    updateStats(data.stats);
                    break;
                case 'done':
                    updateStatus('Goal completed!', 'completed');
                    startBtn.disabled = false;
                    break;
                case 'error':
                    updateStatus('Error: ' + data.message, 'failed');
                    startBtn.disabled = false;
                    break;
                case 'log':
                    addLog(data.message);
                    break;
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
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        # Create orchestrator with WebSocket callback
        client = LLMClient()
        orchestrator = Orchestrator(goal=goal, client=client)
        active_orchestrator = orchestrator
        
        # Override methods to send updates via WebSocket
        original_cycle = orchestrator._cycle
        
        async def cycle_with_updates() -> None:
            await websocket.send_json({"type": "cycle_start", "cycle": orchestrator.cycles + 1})
            
            # Planning
            with_status = orchestrator._cycle.__self__.__class__.__dict__.get('_cycle')
            plan = await orchestrator.planner.plan(orchestrator.goal, orchestrator.cycle_plans, orchestrator.all_tasks)
            
            await websocket.send_json({"type": "plan", "plan": plan.model_dump()})
            
            if not plan.tasks:
                await websocket.send_json({"type": "log", "message": "No tasks planned. Goal may be complete."})
                return
            
            orchestrator.cycle_plans.append(plan)
            for task in plan.tasks:
                orchestrator.all_tasks.append(task)
            
            # Execute tasks
            for task in plan.tasks:
                await websocket.send_json({"type": "task_start", "task": task.model_dump()})
                try:
                    result = await WorkerAgent(task=task, client=orchestrator.client, settings=orchestrator.settings).run()
                    task.status = "completed"
                    task.result = result
                    await websocket.send_json({"type": "task_complete", "task": task.model_dump()})
                except Exception as e:
                    task.status = "failed"
                    task.result = str(e)
                    await websocket.send_json({"type": "task_failed", "task": task.model_dump(), "error": str(e)})
            
            # Test
            test_result = await TesterAgent(client=orchestrator.client, settings=orchestrator.settings).run(orchestrator.goal, orchestrator.all_tasks)
            await websocket.send_json({"type": "test_result", "result": test_result.model_dump()})
            
            if test_result.passed:
                await websocket.send_json({"type": "log", "message": f"Tests passed: {test_result.summary}"})
            else:
                await websocket.send_json({"type": "log", "message": f"Tests failed: {test_result.summary}"})
                for failure in test_result.failures:
                    await websocket.send_json({"type": "log", "message": f"  • {failure}"})
                orchestrator.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            
            # Send cycle summary
            cycle_data = {
                "number": orchestrator.cycles,
                "goal": orchestrator.goal,
                "plan": plan.model_dump(),
                "test_result": test_result.model_dump(),
            }
            
            completed = sum(1 for t in orchestrator.all_tasks if t.status == "completed")
            failed = sum(1 for t in orchestrator.all_tasks if t.status == "failed")
            pending = sum(1 for t in orchestrator.all_tasks if t.status == "pending")
            
            stats = {
                "cycles": orchestrator.cycles,
                "completed": completed,
                "failed": failed,
                "pending": pending,
            }
            
            await websocket.send_json({"type": "cycle_complete", "cycle": orchestrator.cycles, "cycle_data": cycle_data, "stats": stats})
        
        # Run the orchestrator with WebSocket updates
        orchestrator._cycle = cycle_with_updates
        await orchestrator.run()
        
        await websocket.send_json({"type": "done"})
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        active_orchestrator = None


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
