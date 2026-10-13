from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.core.orchestrator import Orchestrator
from furrow.state import get_state_manager


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    workspace: str = "."
    resume: bool = False
    max_cycles: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Furrow", lifespan=lifespan)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #4CAF50; margin-bottom: 20px; }
        .input-group { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        input[type="text"] { flex: 1; min-width: 300px; padding: 12px; border: 1px solid #333; border-radius: 4px; background: #16213e; color: #eee; font-size: 16px; }
        button { padding: 12px 24px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; font-weight: 600; }
        button:hover { background: #45a049; }
        button:disabled { background: #666; cursor: not-allowed; }
        .status-bar { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
        .status-item { padding: 10px 16px; background: #16213e; border-radius: 4px; border-left: 4px solid #4CAF50; }
        .status-item.error { border-left-color: #f44336; }
        .status-item.warning { border-left-color: #ff9800; }
        .status-label { font-size: 12px; color: #888; text-transform: uppercase; }
        .status-value { font-size: 18px; font-weight: 600; }
        .cycles-container { display: grid; gap: 16px; }
        .cycle-card { background: #16213e; border-radius: 8px; border: 1px solid #333; overflow: hidden; }
        .cycle-header { padding: 16px; background: #0f0f23; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }
        .cycle-title { font-size: 18px; font-weight: 600; }
        .cycle-meta { display: flex; gap: 16px; font-size: 14px; color: #888; }
        .cycle-content { padding: 16px; }
        .task-list { display: grid; gap: 8px; margin-top: 12px; }
        .task { padding: 12px; background: #0f0f23; border-radius: 4px; border-left: 4px solid #888; display: flex; gap: 12px; align-items: flex-start; }
        .task.completed { border-left-color: #4CAF50; }
        .task.failed { border-left-color: #f44336; }
        .task.pending { border-left-color: #ff9800; }
        .task-id { font-family: monospace; font-size: 14px; color: #888; min-width: 40px; }
        .task-desc { flex: 1; }
        .task-result { font-size: 13px; color: #aaa; margin-top: 8px; white-space: pre-wrap; font-family: monospace; }
        .test-result { padding: 16px; margin-top: 16px; border-radius: 4px; }
        .test-result.passed { background: #1b5e20; border: 1px solid #4CAF50; }
        .test-result.failed { background: #b71c1c; border: 1px solid #f44336; }
        .test-summary { font-weight: 600; margin-bottom: 8px; }
        .test-failures { font-size: 13px; font-family: monospace; white-space: pre-wrap; }
        .rationale { color: #888; font-style: italic; margin-bottom: 12px; }
        .log-entry { font-family: monospace; font-size: 12px; color: #888; padding: 4px 0; }
        .log-entry.error { color: #f44336; }
        .log-entry.warn { color: #ff9800; }
        .log-entry.info { color: #4CAF50; }
        .empty-state { text-align: center; color: #666; padding: 60px 20px; }
        .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #333; border-top-color: #4CAF50; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 8px; vertical-align: middle; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>Furrow <span id="version">v0.1.0</span></h1>
        <div class="input-group">
            <input type="text" id="goal" placeholder="Enter your goal (e.g., 'Add JWT auth to the API')" />
            <button id="startBtn" onclick="startFurrow()">Start</button>
            <button id="resumeBtn" onclick="startFurrow(true)" style="background: #2196F3;">Resume</button>
            <button id="stopBtn" onclick="stopFurrow()" disabled style="background: #f44336;">Stop</button>
        </div>
        <div class="status-bar">
            <div class="status-item">
                <div class="status-label">Status</div>
                <div class="status-value" id="status">Idle</div>
            </div>
            <div class="status-item">
                <div class="status-label">Cycles</div>
                <div class="status-value" id="cycleCount">0</div>
            </div>
            <div class="status-item">
                <div class="status-label">Completed Tasks</div>
                <div class="status-value" id="completedCount">0</div>
            </div>
            <div class="status-item">
                <div class="status-label">Failed Tasks</div>
                <div class="status-value" id="failedCount">0</div>
            </div>
        </div>
        <div id="emptyState" class="empty-state">
            <p>Enter a goal and click Start to begin the Furrow development loop.</p>
        </div>
        <div id="cyclesContainer" class="cycles-container" style="display: none;"></div>
    </div>
    <script>
        let ws = null;
        let currentCycle = 0;

        function startFurrow(resume = false) {
            const goal = document.getElementById('goal').value.trim();
            if (!goal && !resume) {
                alert('Please enter a goal');
                return;
            }

            document.getElementById('startBtn').disabled = true;
            document.getElementById('resumeBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('emptyState').style.display = 'none';
            document.getElementById('cyclesContainer').style.display = 'grid';
            document.getElementById('cyclesContainer').innerHTML = '';

            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onopen = () => {
                ws.send(JSON.stringify({goal, resume, model: null, workspace: '.', max_cycles: 0}));
                updateStatus('Running...');
            };
            ws.onmessage = handleMessage;
            ws.onclose = () => {
                updateStatus('Disconnected');
                document.getElementById('startBtn').disabled = false;
                document.getElementById('resumeBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
            };
        }

        function stopFurrow() {
            if (ws) {
                ws.close();
            }
        }

        function handleMessage(event) {
            try {
                const msg = JSON.parse(event.data);
                switch (msg.type) {
                    case 'cycle_start':
                        addCycle(msg);
                        break;
                    case 'planning':
                        updateCycle(msg.cycle, 'planning', msg);
                        break;
                    case 'plan':
                        updateCyclePlan(msg.cycle, msg.plan);
                        break;
                    case 'executing':
                        updateCycle(msg.cycle, 'executing', msg);
                        break;
                    case 'task_result':
                        updateTask(msg.cycle, msg.task);
                        break;
                    case 'testing':
                        updateCycle(msg.cycle, 'testing', msg);
                        break;
                    case 'test_result':
                        updateTestResult(msg.cycle, msg.result);
                        break;
                    case 'cycle_complete':
                        completeCycle(msg.cycle);
                        break;
                    case 'done':
                        updateStatus('Complete');
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('resumeBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                        break;
                    case 'error':
                        addLog(msg.cycle, msg.message, 'error');
                        break;
                    case 'log':
                        addLog(msg.cycle, msg.message, msg.level);
                        break;
                }
            } catch (e) {
                console.error('Failed to parse message:', e);
            }
        }

        function addCycle(msg) {
            currentCycle = msg.cycle;
            const container = document.getElementById('cyclesContainer');
            const card = document.createElement('div');
            card.className = 'cycle-card';
            card.id = 'cycle-' + msg.cycle;
            card.innerHTML = `
                <div class="cycle-header">
                    <div class="cycle-title">Cycle <span class="cycle-num">${msg.cycle}</span> <span class="spinner"></span></div>
                    <div class="cycle-meta">
                        <span>Goal: <span id="cycle-${msg.cycle}-goal">${escapeHtml(msg.goal)}</span></span>
                        <span>Status: <span id="cycle-${msg.cycle}-status" class="pending">Planning</span></span>
                    </div>
                </div>
                <div class="cycle-content">
                    <div class="rationale" id="cycle-${msg.cycle}-rationale"></div>
                    <div class="task-list" id="cycle-${msg.cycle}-tasks"></div>
                </div>
            `;
            container.prepend(card);
            updateStatus(`Cycle ${msg.cycle} - Planning`);
            document.getElementById('cycleCount').textContent = msg.cycle;
        }

        function updateCycle(cycle, phase, msg) {
            const statusEl = document.getElementById(`cycle-${cycle}-status`);
            const spinner = document.querySelector(`#cycle-${cycle} .spinner`);
            if (statusEl) {
                statusEl.textContent = phase.charAt(0).toUpperCase() + phase.slice(1);
                statusEl.className = phase;
            }
            if (phase === 'planning' || phase === 'executing' || phase === 'testing') {
                if (spinner) spinner.style.display = 'inline-block';
            } else {
                if (spinner) spinner.style.display = 'none';
            }
            updateStatus(`Cycle ${cycle} - ${phase.charAt(0).toUpperCase() + phase.slice(1)}`);
        }

        function updateCyclePlan(cycle, plan) {
            const rationaleEl = document.getElementById(`cycle-${cycle}-rationale`);
            if (rationaleEl && plan.rationale) {
                rationaleEl.textContent = plan.rationale;
            }
            const tasksEl = document.getElementById(`cycle-${cycle}-tasks`);
            if (tasksEl && plan.tasks) {
                tasksEl.innerHTML = plan.tasks.map(task => `
                    <div class="task pending" id="task-${cycle}-${task.id}">
                        <span class="task-id">${task.id}</span>
                        <div class="task-desc">${escapeHtml(task.description)}</div>
                    </div>
                `).join('');
            }
        }

        function updateTask(cycle, task) {
            const taskEl = document.getElementById(`task-${cycle}-${task.id}`);
            if (taskEl) {
                taskEl.className = 'task ' + task.status;
                if (task.result) {
                    let resultEl = taskEl.querySelector('.task-result');
                    if (!resultEl) {
                        resultEl = document.createElement('div');
                        resultEl.className = 'task-result';
                        taskEl.appendChild(resultEl);
                    }
                    resultEl.textContent = task.result;
                }
            }
            updateCounts();
        }

        function updateTestResult(cycle, result) {
            const contentEl = document.getElementById(`cycle-${cycle}`)?.querySelector('.cycle-content');
            if (!contentEl) return;

            let testEl = contentEl.querySelector('.test-result');
            if (!testEl) {
                testEl = document.createElement('div');
                testEl.className = 'test-result';
                contentEl.appendChild(testEl);
            }
            testEl.className = 'test-result ' + (result.passed ? 'passed' : 'failed');
            testEl.innerHTML = `
                <div class="test-summary">${result.passed ? '✓ Tests Passed' : '✗ Tests Failed'}: ${escapeHtml(result.summary)}</div>
                ${result.failures.length > 0 ? `<div class="test-failures">${result.failures.map(f => '• ' + escapeHtml(f)).join('\\n')}</div>` : ''}
            `;
            updateCounts();
        }

        function completeCycle(cycle) {
            const statusEl = document.getElementById(`cycle-${cycle}-status`);
            const spinner = document.querySelector(`#cycle-${cycle} .spinner`);
            if (statusEl) {
                statusEl.textContent = 'Complete';
                statusEl.className = 'completed';
            }
            if (spinner) spinner.style.display = 'none';
        }

        function updateCounts() {
            const tasks = document.querySelectorAll('.task');
            let completed = 0, failed = 0;
            tasks.forEach(t => {
                if (t.classList.contains('completed')) completed++;
                if (t.classList.contains('failed')) failed++;
            });
            document.getElementById('completedCount').textContent = completed;
            document.getElementById('failedCount').textContent = failed;
        }

        function updateStatus(status) {
            document.getElementById('status').textContent = status;
        }

        function addLog(cycle, message, level = 'info') {
            const contentEl = document.getElementById(`cycle-${cycle}`)?.querySelector('.cycle-content');
            if (!contentEl) return;
            let logEl = contentEl.querySelector('.log-container');
            if (!logEl) {
                logEl = document.createElement('div');
                logEl.className = 'log-container';
                logEl.style.marginTop = '12px';
                logEl.style.maxHeight = '200px';
                logEl.style.overflow = 'auto';
                contentEl.appendChild(logEl);
            }
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + level;
            entry.textContent = message;
            logEl.appendChild(entry);
            logEl.scrollTop = logEl.scrollHeight;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=HTML_TEMPLATE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        workspace = data.get("workspace", ".")
        resume = data.get("resume", False)
        max_cycles = data.get("max_cycles", 0)

        if model:
            from furrow.config import settings
            settings.model = model
        if max_cycles > 0:
            from furrow.config import settings
            settings.max_cycles = max_cycles

        state_manager = get_state_manager(workspace)
        orchestrator = Orchestrator(goal=goal, workspace=workspace, state_manager=state_manager, resume=resume)

        async def send_message(msg: dict[str, Any]) -> None:
            try:
                await websocket.send_text(json.dumps(msg))
            except (RuntimeError, WebSocketDisconnect, OSError):
                pass

        # Monkey-patch the orchestrator to send WebSocket updates

        async def patched_cycle() -> None:
            await send_message({"type": "cycle_start", "cycle": orchestrator.cycles + 1, "goal": orchestrator.goal})

            # Planning
            await send_message({"type": "planning", "cycle": orchestrator.cycles + 1})
            plan = await orchestrator.planner.plan(orchestrator.goal)
            await send_message({"type": "plan", "cycle": orchestrator.cycles + 1, "plan": plan.model_dump()})

            if not plan.tasks:
                await send_message({"type": "log", "cycle": orchestrator.cycles + 1, "message": "No tasks planned. Goal may be complete.", "level": "warn"})
                return

            orchestrator.state_manager.start_cycle(orchestrator.goal, plan.rationale, plan.tasks)

            # Executing
            await send_message({"type": "executing", "cycle": orchestrator.cycles + 1})
            worker_tasks = [
                WorkerAgent(task=task, client=orchestrator.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            for task, result in zip(plan.tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    await send_message({"type": "task_result", "cycle": orchestrator.cycles + 1, "task": {"id": task.id, "status": "failed", "result": str(result)}})
                    orchestrator.state_manager.update_task_result(task.id, "failed", str(result))
                else:
                    task.status = "completed"
                    task.result = result
                    await send_message({"type": "task_result", "cycle": orchestrator.cycles + 1, "task": {"id": task.id, "status": "completed", "result": result}})
                    orchestrator.state_manager.update_task_result(task.id, "completed", result)

            # Testing
            await send_message({"type": "testing", "cycle": orchestrator.cycles + 1})
            test_result = await TesterAgent(client=orchestrator.client).run(orchestrator.goal, plan.tasks)

            orchestrator.state_manager.set_test_result(
                test_result.passed,
                test_result.summary,
                test_result.failures
            )

            await send_message({"type": "test_result", "cycle": orchestrator.cycles + 1, "result": test_result.model_dump()})

            if test_result.passed:
                await send_message({"type": "log", "cycle": orchestrator.cycles + 1, "message": f"Tests passed: {test_result.summary}", "level": "info"})
            else:
                await send_message({"type": "log", "cycle": orchestrator.cycles + 1, "message": f"Tests failed: {test_result.summary}", "level": "error"})
                for failure in test_result.failures:
                    await send_message({"type": "log", "cycle": orchestrator.cycles + 1, "message": f"  • {failure}", "level": "error"})
                orchestrator.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

            await send_message({"type": "cycle_complete", "cycle": orchestrator.cycles + 1})

        orchestrator._cycle = patched_cycle
        await orchestrator.run()
        await send_message({"type": "done"})

    except (RuntimeError, WebSocketDisconnect, OSError, ValueError) as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except (RuntimeError, WebSocketDisconnect, OSError):
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)