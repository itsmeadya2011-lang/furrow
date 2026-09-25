from __future__ import annotations

import asyncio
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, Provider
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None
    provider: Optional[str] = None
    max_cycles: Optional[int] = None


# Store active orchestrators for status polling
active_orchestrators: dict[str, Orchestrator] = {}


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow - Autonomous Coding Agent</title>
    <style>
        * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { margin: 0; padding: 20px; background: #1a1a2e; color: #eaeaea; min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00d9ff; margin-bottom: 5px; }
        .subtitle { color: #888; margin-bottom: 30px; }
        .panel { background: #16213e; border-radius: 8px; padding: 20px; margin-bottom: 20px; border: 1px solid #0f3460; }
        .panel h2 { margin-top: 0; color: #00d9ff; font-size: 1.2rem; }
        .form-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 15px; }
        .form-row > * { flex: 1; min-width: 200px; }
        input, select { padding: 10px; border-radius: 4px; border: 1px solid #0f3460; background: #0f3460; color: #eaeaea; font-size: 14px; }
        input:focus, select:focus { outline: none; border-color: #00d9ff; }
        button { padding: 12px 24px; border-radius: 4px; border: none; background: #00d9ff; color: #1a1a2e; font-weight: 600; cursor: pointer; font-size: 14px; }
        button:hover { background: #00b8d4; }
        button:disabled { background: #555; cursor: not-allowed; }
        .status { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
        .stat { background: #0f3460; padding: 15px 20px; border-radius: 8px; min-width: 120px; }
        .stat-value { font-size: 2rem; font-weight: bold; color: #00d9ff; }
        .stat-label { font-size: 0.85rem; color: #888; }
        .task-list { max-height: 400px; overflow-y: auto; }
        .task { background: #0f3460; border-radius: 6px; padding: 12px 16px; margin-bottom: 8px; border-left: 4px solid #555; transition: all 0.2s; }
        .task.pending { border-left-color: #f39c12; }
        .task.running { border-left-color: #3498db; animation: pulse 1.5s infinite; }
        .task.completed { border-left-color: #27ae60; }
        .task.failed { border-left-color: #e74c3c; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .task-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .task-id { font-weight: bold; color: #00d9ff; font-size: 0.9rem; }
        .task-status { font-size: 0.75rem; padding: 2px 8px; border-radius: 12px; text-transform: uppercase; }
        .task-status.pending { background: #f39c12; color: #1a1a2e; }
        .task-status.running { background: #3498db; color: white; }
        .task-status.completed { background: #27ae60; color: white; }
        .task-status.failed { background: #e74c3c; color: white; }
        .task-desc { color: #ccc; font-size: 0.9rem; }
        .task-files { font-size: 0.75rem; color: #888; margin-top: 4px; font-family: monospace; }
        .task-result { font-size: 0.8rem; color: #aaa; margin-top: 8px; padding: 8px; background: #1a1a2e; border-radius: 4px; white-space: pre-wrap; max-height: 100px; overflow: auto; }
        .log { background: #0f3460; border-radius: 6px; padding: 15px; font-family: monospace; font-size: 0.8rem; max-height: 300px; overflow-y: auto; color: #aaa; }
        .log-line { margin: 2px 0; }
        .log-line.info { color: #3498db; }
        .log-line.success { color: #27ae60; }
        .log-line.error { color: #e74c3c; }
        .log-line.warning { color: #f39c12; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Furrow</h1>
        <p class="subtitle">Autonomous coding agent with infinite parallel development loop</p>

        <div class="panel" id="setup-panel">
            <h2>New Session</h2>
            <div class="form-row">
                <input type="text" id="goal" placeholder="Enter your goal (e.g., 'Add JWT authentication to the API')" />
                <select id="provider">
                    <option value="anthropic">Anthropic (Claude)</option>
                    <option value="openai">OpenAI (GPT)</option>
                    <option value="ollama">Ollama (Local)</option>
                </select>
            </div>
            <div class="form-row">
                <input type="text" id="model" placeholder="Model override (optional)" />
                <input type="number" id="max-cycles" placeholder="Max cycles (0 = unlimited)" value="0" style="width: 100px;" />
            </div>
            <button id="start-btn" onclick="startSession()">Start Furrow</button>
        </div>

        <div class="panel hidden" id="status-panel">
            <h2>Session Status</h2>
            <div class="status">
                <div class="stat">
                    <div class="stat-value" id="cycle-count">0</div>
                    <div class="stat-label">Cycles</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="task-total">0</div>
                    <div class="stat-label">Total Tasks</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="task-completed">0</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="task-failed">0</div>
                    <div class="stat-label">Failed</div>
                </div>
            </div>
            <div class="task-list" id="task-list"></div>
        </div>

        <div class="panel">
            <h2>Activity Log</h2>
            <div class="log" id="log"></div>
        </div>
    </div>

    <script>
        let ws = null;
        let sessionId = null;

        function log(message, type = 'info') {
            const logEl = document.getElementById('log');
            const line = document.createElement('div');
            line.className = `log-line ${type}`;
            line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logEl.appendChild(line);
            logEl.scrollTop = logEl.scrollHeight;
        }

        function updateTaskList(tasks) {
            const container = document.getElementById('task-list');
            container.innerHTML = '';
            let completed = 0, failed = 0, running = 0;
            tasks.forEach(task => {
                const el = document.createElement('div');
                el.className = `task ${task.status}`;
                if (task.status === 'completed') completed++;
                if (task.status === 'failed') failed++;
                if (task.status === 'running') running++;
                el.innerHTML = `
                    <div class="task-header">
                        <span class="task-id">Task ${task.id}</span>
                        <span class="task-status ${task.status}">${task.status}</span>
                    </div>
                    <div class="task-desc">${task.description}</div>
                    ${task.files && task.files.length ? `<div class="task-files">Files: ${task.files.join(', ')}</div>` : ''}
                    ${task.result ? `<div class="task-result">${task.result}</div>` : ''}
                `;
                container.appendChild(el);
            });
            document.getElementById('task-total').textContent = tasks.length;
            document.getElementById('task-completed').textContent = completed;
            document.getElementById('task-failed').textContent = failed;
        }

        function startSession() {
            const goal = document.getElementById('goal').value.trim();
            if (!goal) return alert('Please enter a goal');

            const provider = document.getElementById('provider').value;
            const model = document.getElementById('model').value.trim() || null;
            const maxCycles = parseInt(document.getElementById('max-cycles').value) || 0;

            document.getElementById('setup-panel').classList.add('hidden');
            document.getElementById('status-panel').classList.remove('hidden');
            log(`Starting session with goal: ${goal}`);
            log(`Provider: ${provider}${model ? ', Model: ' + model : ''}${maxCycles ? ', Max cycles: ' + maxCycles : ''}`);

            sessionId = 'session_' + Date.now();
            ws = new WebSocket(`ws://${location.host}/ws/${sessionId}`);

            ws.onopen = () => {
                ws.send(JSON.stringify({ goal, provider, model, max_cycles: maxCycles }));
            };

            ws.onmessage = (ev) => {
                try {
                    const data = JSON.parse(ev.data);
                    handleMessage(data);
                } catch (e) {
                    log(ev.data, 'info');
                }
            };

            ws.onclose = () => {
                log('WebSocket connection closed', 'warning');
            };

            ws.onerror = (err) => {
                log('WebSocket error: ' + err.message, 'error');
            };
        }

        function handleMessage(data) {
            if (data.type === 'cycle_start') {
                document.getElementById('cycle-count').textContent = data.cycle;
                log(`═══ Cycle ${data.cycle} ═══`, 'info');
            } else if (data.type === 'plan') {
                log(`Plan: ${data.rationale}`, 'info');
                updateTaskList(data.tasks);
            } else if (data.type === 'task_start') {
                log(`Starting task ${data.task_id}: ${data.description}`, 'info');
                // Update task status to running
                const tasks = getCurrentTasks();
                const task = tasks.find(t => t.id === data.task_id);
                if (task) {
                    task.status = 'running';
                    updateTaskList(tasks);
                }
            } else if (data.type === 'task_complete') {
                log(`Task ${data.task_id} completed`, 'success');
                const tasks = getCurrentTasks();
                const task = tasks.find(t => t.id === data.task_id);
                if (task) {
                    task.status = 'completed';
                    task.result = data.result;
                    updateTaskList(tasks);
                }
            } else if (data.type === 'task_failed') {
                log(`Task ${data.task_id} failed: ${data.error}`, 'error');
                const tasks = getCurrentTasks();
                const task = tasks.find(t => t.id === data.task_id);
                if (task) {
                    task.status = 'failed';
                    task.result = data.error;
                    updateTaskList(tasks);
                }
            } else if (data.type === 'test_result') {
                if (data.passed) {
                    log(`Tests passed: ${data.summary}`, 'success');
                } else {
                    log(`Tests failed: ${data.summary}`, 'error');
                    data.failures.forEach(f => log(`  - ${f}`, 'error'));
                }
            } else if (data.type === 'done') {
                log('Goal complete! 🎉', 'success');
                ws.close();
            } else if (data.type === 'error') {
                log(`Error: ${data.message}`, 'error');
            } else if (data.type === 'log') {
                log(data.message, data.level || 'info');
            }
        }

        function getCurrentTasks() {
            // Reconstruct from DOM
            const tasks = [];
            document.querySelectorAll('#task-list .task').forEach(el => {
                const id = el.querySelector('.task-id').textContent.replace('Task ', '');
                const status = el.querySelector('.task-status').textContent.toLowerCase();
                const desc = el.querySelector('.task-desc').textContent;
                const filesEl = el.querySelector('.task-files');
                const files = filesEl ? filesEl.textContent.replace('Files: ', '').split(', ') : [];
                const resultEl = el.querySelector('.task-result');
                const result = resultEl ? resultEl.textContent : null;
                tasks.push({ id, status, description: desc, files, result });
            });
            return tasks;
        }
    </script>
</body>
</html>
""")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    orchestrator = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        provider = data.get("provider")
        model = data.get("model")
        max_cycles = data.get("max_cycles", 0)

        config = Settings()
        if provider:
            config.provider = Provider(provider)
        if model:
            config.model = model
            config.planner_model = model
            config.worker_model = model
            config.tester_model = model
        if max_cycles:
            config.max_cycles = max_cycles

        # Patch orchestrator to send WebSocket updates
        class WebSocketOrchestrator(Orchestrator):
            async def _send(self, msg: dict) -> None:
                await websocket.send_json(msg)

            async def run(self) -> None:
                await self._send({"type": "log", "message": f"Starting Furrow with goal: {self.goal}", "level": "info"})
                self._load_state()
                while True:
                    if self.config.max_cycles > 0 and self.cycles >= self.config.max_cycles:
                        await self._send({"type": "log", "message": f"Max cycles ({self.config.max_cycles}) reached.", "level": "warning"})
                        break
                    self.cycles += 1
                    await self._send({"type": "cycle_start", "cycle": self.cycles})
                    await self._cycle()
                    self._save_state()
                    if self._is_done():
                        await self._send({"type": "done"})
                        break

            async def _cycle(self) -> None:
                await self._send({"type": "log", "message": "Planning...", "level": "info"})
                plan = await self.planner.plan(self.goal, self.all_tasks)
                await self._send({
                    "type": "plan",
                    "rationale": plan.rationale,
                    "tasks": [t.model_dump() for t in plan.tasks],
                })

                if not plan.tasks:
                    await self._send({"type": "log", "message": "No tasks planned. Goal may be complete.", "level": "warning"})
                    return

                for task in plan.tasks:
                    if not any(t.id == task.id for t in self.all_tasks):
                        self.all_tasks.append(task)

                # Run tasks with individual status updates
                for task in plan.tasks:
                    await self._send({"type": "task_start", "task_id": task.id, "description": task.description})
                    try:
                        worker = WorkerAgent(task=task, client=self.client, settings=self.config)
                        result = await worker.run()
                        task.status = "completed"
                        task.result = result
                        await self._send({"type": "task_complete", "task_id": task.id, "result": result})
                    except Exception as e:
                        task.status = "failed"
                        task.result = str(e)
                        await self._send({"type": "task_failed", "task_id": task.id, "error": str(e)})

                await self._send({"type": "log", "message": "Running tests...", "level": "info"})
                test_result = await TesterAgent(client=self.client, settings=self.config).run(self.goal, plan.tasks)

                if test_result.passed:
                    await self._send({"type": "test_result", "passed": True, "summary": test_result.summary, "failures": []})
                else:
                    await self._send({"type": "test_result", "passed": False, "summary": test_result.summary, "failures": test_result.failures})
                    self.goal = f"{self.original_goal}\n\nPrevious failures to fix:\n" + "\n".join(test_result.failures)

        from furrow.agents.worker import WorkerAgent
        from furrow.agents.tester import TesterAgent

        orchestrator = WebSocketOrchestrator(goal=goal, config=config)
        active_orchestrators[session_id] = orchestrator
        await orchestrator.run()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if session_id in active_orchestrators:
            del active_orchestrators[session_id]


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
