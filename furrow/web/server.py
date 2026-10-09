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

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


# Store active orchestrator for status updates
active_orchestrator: Optional[Orchestrator] = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #4CAF50; }
        .form-group { margin-bottom: 15px; }
        input[type="text"] { width: 70%; padding: 10px; background: #16213e; border: 1px solid #0f3460; color: #eee; border-radius: 4px; }
        button { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        button:disabled { background: #666; cursor: not-allowed; }
        #out { background: #0f0f23; padding: 15px; border-radius: 4px; max-height: 600px; overflow-y: auto; font-family: monospace; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }
        .cycle { border-left: 3px solid #4CAF50; padding-left: 10px; margin: 10px 0; }
        .task { margin: 5px 0; padding: 5px 10px; border-radius: 3px; }
        .task.pending { background: #0f3460; }
        .task.completed { background: #1b5e20; }
        .task.failed { background: #b71c1c; }
        .status-bar { display: flex; gap: 20px; margin-bottom: 15px; flex-wrap: wrap; }
        .status-item { background: #16213e; padding: 10px 15px; border-radius: 4px; }
        .status-label { color: #888; font-size: 12px; }
        .status-value { font-weight: bold; font-size: 18px; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <div class="status-bar">
        <div class="status-item"><div class="status-label">Cycle</div><div class="status-value" id="cycle">0</div></div>
        <div class="status-item"><div class="status-label">Tasks</div><div class="status-value" id="task-count">0</div></div>
        <div class="status-item"><div class="status-label">Completed</div><div class="status-value" id="completed-count">0</div></div>
        <div class="status-item"><div class="status-label">Failed</div><div class="status-value" id="failed-count">0</div></div>
    </div>
    <form id="form">
        <div class="form-group">
            <input id="goal" type="text" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
            <button type="submit" id="start-btn">Start</button>
        </div>
    </form>
    <div id="out"></div>
    <script>
        const form = document.getElementById('form');
        const out = document.getElementById('out');
        const startBtn = document.getElementById('start-btn');
        let ws = null;
        let taskStates = {};

        function updateStatus(data) {
            if (data.cycle !== undefined) document.getElementById('cycle').textContent = data.cycle;
            if (data.tasks !== undefined) document.getElementById('task-count').textContent = data.tasks;
            if (data.completed !== undefined) document.getElementById('completed').textContent = data.completed;
            if (data.failed !== undefined) document.getElementById('failed').textContent = data.failed;
        }

        function renderTasks() {
            const tasksDiv = document.getElementById('tasks');
            if (!tasksDiv) return;
            let html = '<h3>Tasks</h3>';
            for (const [id, task] of Object.entries(taskStates)) {
                html += `<div class="task ${task.status}">[${task.status}] ${task.id}: ${task.description}</div>`;
            }
            tasksDiv.innerHTML = html;
        }

        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value;
            out.textContent += '\\nStarting: ' + goal + '\\n';
            startBtn.disabled = true;
            taskStates = {};

            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                try {
                    const data = JSON.parse(ev.data);
                    if (data.type === 'status') {
                        updateStatus(data);
                    } else if (data.type === 'task') {
                        taskStates[data.task.id] = data.task;
                        renderTasks();
                    } else if (data.type === 'log') {
                        out.textContent += data.message + '\\n';
                        out.scrollTop = out.scrollHeight;
                    } else if (data.type === 'done') {
                        out.textContent += '\\n=== DONE ===' + '\\n';
                        startBtn.disabled = false;
                    }
                } catch {
                    out.textContent += ev.data + '\\n';
                    out.scrollTop = out.scrollHeight;
                }
            };
            ws.onclose = () => {
                out.textContent += '\\nConnection closed.\\n';
                startBtn.disabled = false;
            };
            ws.onopen = () => {
                ws.send(JSON.stringify({goal: goal}));
            };
        };
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
        if not goal:
            await websocket.send_json({"type": "log", "message": "Error: No goal provided"})
            return

        orchestrator = Orchestrator(goal=goal)
        active_orchestrator = orchestrator

        # Monkey-patch to send updates via WebSocket
        original_cycle = orchestrator._cycle

        async def cycle_with_ws():
            await original_cycle()
            # Send status update
            completed = sum(1 for t in orchestrator.all_tasks if t.status == "completed")
            failed = sum(1 for t in orchestrator.all_tasks if t.status == "failed")
            pending = sum(1 for t in orchestrator.all_tasks if t.status == "pending")
            await websocket.send_json({
                "type": "status",
                "cycle": orchestrator.cycles,
                "tasks": len(orchestrator.all_tasks),
                "completed": completed,
                "failed": failed,
                "pending": pending,
            })
            # Send task updates
            for task in orchestrator.all_tasks:
                await websocket.send_json({
                    "type": "task",
                    "task": {
                        "id": task.id,
                        "description": task.description,
                        "status": task.status,
                        "result": task.result,
                    }
                })

        orchestrator._cycle = cycle_with_ws

        # Also patch console.print to send logs
        import furrow.core.orchestrator as orch_module
        original_print = orch_module.console.print

        def ws_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            asyncio.create_task(websocket.send_json({"type": "log", "message": msg}))
            original_print(*args, **kwargs)

        orch_module.console.print = ws_print

        await orchestrator.run()

        await websocket.send_json({"type": "done", "message": "Goal completed"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "log", "message": f"Error: {e}"})
    finally:
        active_orchestrator = None


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
