from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

console = Console()

# Global state for web UI
active_orchestrator: Orchestrator | None = None


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #4CAF50; }
        .form-group { margin-bottom: 15px; }
        input[type="text"] { width: 70%; padding: 10px; background: #16213e; border: 1px solid #0f3460; color: #eee; border-radius: 4px; }
        button { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        button:disabled { background: #666; cursor: not-allowed; }
        #output { background: #0f0f23; padding: 15px; border-radius: 4px; height: 400px; overflow-y: auto; font-family: monospace; font-size: 13px; white-space: pre-wrap; border: 1px solid #0f3460; }
        .cycle { color: #2196F3; font-weight: bold; margin-top: 10px; }
        .task { margin-left: 20px; margin-top: 5px; }
        .task.pending { color: #FFC107; }
        .task.in_progress { color: #2196F3; }
        .task.completed { color: #4CAF50; }
        .task.failed { color: #f44336; }
        .status-bar { display: flex; gap: 20px; margin-bottom: 15px; padding: 10px; background: #16213e; border-radius: 4px; }
        .stat { font-weight: bold; }
    </style>
</head>
<body>
    <h1>Furrow - Autonomous Coding Agent</h1>
    <div class="form-group">
        <input id="goal" type="text" placeholder="Enter your goal (e.g., 'Add JWT auth to the API')" />
        <button id="startBtn" onclick="start()">Start</button>
        <button id="stopBtn" onclick="stop()" disabled>Stop</button>
    </div>
    <div class="status-bar">
        <div class="stat">Cycles: <span id="cycles">0</span></div>
        <div class="stat">Tasks: <span id="taskCount">0</span></div>
        <div class="stat">Completed: <span id="completedCount">0</span></div>
        <div class="stat">Failed: <span id="failedCount">0</span></div>
    </div>
    <div id="output"></div>
    <script>
        let ws = null;
        const output = document.getElementById('output');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const goalInput = document.getElementById('goal');

        function log(msg) {
            output.textContent += msg + '\\n';
            output.scrollTop = output.scrollHeight;
        }

        function updateStats(data) {
            if (data.cycles !== undefined) document.getElementById('cycles').textContent = data.cycles;
            if (data.tasks !== undefined) document.getElementById('taskCount').textContent = data.tasks;
            if (data.completed !== undefined) document.getElementById('completedCount').textContent = data.completed;
            if (data.failed !== undefined) document.getElementById('failedCount').textContent = data.failed;
        }

        function renderTasks(tasks) {
            if (!tasks) return;
            // This is called via a dedicated message type
        }

        function start() {
            const goal = goalInput.value.trim();
            if (!goal) return alert('Enter a goal');
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            goalInput.disabled = true;
            log('\\n[Starting Furrow...]\\n');
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                try {
                    const data = JSON.parse(ev.data);
                    if (data.type === 'log') {
                        log(data.message);
                    } else if (data.type === 'stats') {
                        updateStats(data);
                    } else if (data.type === 'tasks') {
                        // Re-render task list
                        log(data.tasks.map(t => `  [${t.status.toUpperCase()}] ${t.id}: ${t.description}`).join('\\n'));
                    } else if (data.type === 'done') {
                        log('\\n[Goal complete!]');
                        stop();
                    }
                } catch {
                    log(ev.data);
                }
            };
            ws.onclose = () => {
                log('\\n[Connection closed]');
                stop();
            };
            ws.onopen = () => {
                ws.send(JSON.stringify({goal}));
            };
        }

        function stop() {
            if (ws) {
                ws.close();
                ws = null;
            }
            startBtn.disabled = false;
            stopBtn.disabled = true;
            goalInput.disabled = false;
        }

        goalInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') start();
        });
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
        
        # Custom orchestrator that sends updates via websocket
        class WebOrchestrator(Orchestrator):
            async def _cycle(self) -> None:
                await self._send_stats()
                await super()._cycle()
                await self._send_stats()
            
            async def _send_stats(self):
                completed = sum(1 for t in self.tasks if t.status == "completed")
                failed = sum(1 for t in self.tasks if t.status == "failed")
                await websocket.send_json({
                    "type": "stats",
                    "cycles": self.cycles,
                    "tasks": len(self.tasks),
                    "completed": completed,
                    "failed": failed,
                })
                await websocket.send_json({
                    "type": "tasks",
                    "tasks": [t.model_dump() for t in self.tasks]
                })
            
            def _log(self, message: str):
                asyncio.create_task(websocket.send_json({"type": "log", "message": message}))

        # Monkey-patch console.print to send via websocket
        original_print = console.print
        def ws_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            asyncio.create_task(websocket.send_json({"type": "log", "message": msg}))
            original_print(*args, **kwargs)
        
        import furrow.core.orchestrator as orch_module
        orch_module.console.print = ws_print
        
        active_orchestrator = WebOrchestrator(goal=goal)
        await active_orchestrator.run()
        
        await websocket.send_json({"type": "done"})
        
    except WebSocketDisconnect:
        pass
    except (RuntimeError, OSError, ValueError) as e:
        await websocket.send_json({"type": "log", "message": f"Error: {e}"})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
