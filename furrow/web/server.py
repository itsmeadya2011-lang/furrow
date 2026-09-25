from __future__ import annotations

import asyncio
import json

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Global orchestrator reference for status polling
_current_orchestrator: Orchestrator | None = None


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class WebSocketConsole(Console):
    """Custom console that writes to a WebSocket."""

    def __init__(self, websocket: WebSocket, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.websocket = websocket

    def print(self, *args, **kwargs):
        # Capture the output
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            super().print(*args, **kwargs)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        if output.strip():
            # Send to websocket (non-blocking)
            asyncio.create_task(self._send(output.rstrip()))

    async def _send(self, message: str):
        try:
            await self.websocket.send_text(message)
        except RuntimeError:
            pass  # Connection closed


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
        h1 { color: #4ec9b0; }
        form { margin-bottom: 20px; }
        input { padding: 10px; width: 70%; background: #3c3c3c; border: 1px solid #555; color: #d4d4d4; }
        button { padding: 10px 20px; background: #0e639c; color: white; border: none; cursor: pointer; }
        button:hover { background: #1177bb; }
        button:disabled { background: #555; cursor: not-allowed; }
        pre { background: #252526; padding: 15px; border-radius: 4px; overflow-x: auto; max-height: 70vh; overflow-y: auto; }
        .status { padding: 10px; margin-bottom: 10px; border-radius: 4px; }
        .status.running { background: #0e639c; }
        .status.complete { background: #4ec9b0; }
        .status.error { background: #f44747; }
        .task { margin: 5px 0; padding: 5px 10px; background: #3c3c3c; border-radius: 3px; }
        .task.completed { border-left: 3px solid #4ec9b0; }
        .task.failed { border-left: 3px solid #f44747; }
        .task.pending { border-left: 3px solid #0e639c; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <form id="form">
        <input id="goal" placeholder="Enter goal" required />
        <button type="submit" id="btn">Start</button>
    </form>
    <div id="status" class="status" style="display:none;"></div>
    <div id="tasks"></div>
    <pre id="out"></pre>
    <script>
        const form = document.getElementById('form');
        const out = document.getElementById('out');
        const status = document.getElementById('status');
        const tasks = document.getElementById('tasks');
        const btn = document.getElementById('btn');
        let ws = null;

        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value;
            if (!goal) return;

            out.textContent = '';
            tasks.innerHTML = '';
            status.textContent = 'Connecting...';
            status.className = 'status running';
            status.style.display = 'block';
            btn.disabled = true;

            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                try {
                    const data = JSON.parse(ev.data);
                    if (data.type === 'log') {
                        out.textContent += data.message + '\\n';
                        out.scrollTop = out.scrollHeight;
                    } else if (data.type === 'status') {
                        updateStatus(data);
                    } else if (data.type === 'tasks') {
                        updateTasks(data.tasks);
                    }
                } catch {
                    out.textContent += ev.data + '\\n';
                    out.scrollTop = out.scrollHeight;
                }
            };
            ws.onclose = () => {
                out.textContent += '\\n--- Connection closed ---\\n';
                btn.disabled = false;
                status.textContent = 'Disconnected';
                status.className = 'status';
            };
            ws.onerror = (err) => {
                out.textContent += '\\n--- WebSocket error ---\\n';
            };
            ws.send(JSON.stringify({goal: goal}));
        };

        function updateStatus(data) {
            status.textContent = data.message;
            status.className = 'status ' + (data.done ? 'complete' : 'running');
            if (data.cycles !== undefined) {
                status.textContent = 'Cycle ' + data.cycles + ': ' + data.message;
            }
        }

        function updateTasks(taskList) {
            tasks.innerHTML = '';
            taskList.forEach(task => {
                const div = document.createElement('div');
                div.className = 'task ' + (task.status || 'pending');
                div.textContent = task.id + ': ' + task.description + ' [' + (task.status || 'pending') + ']';
                tasks.appendChild(div);
            });
        }
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    global _current_orchestrator

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        if not goal:
            await websocket.send_text(json.dumps({"type": "log", "message": "Error: No goal provided"}))
            return

        # Create custom console that writes to websocket
        ws_console = WebSocketConsole(websocket)

        # Create orchestrator with custom console
        from furrow.llm import LLMClient

        client = LLMClient()
        orchestrator = Orchestrator(goal=goal, client=client)
        orchestrator.console = ws_console  # type: ignore
        _current_orchestrator = orchestrator

        # Send initial status
        await websocket.send_text(json.dumps({
            "type": "status",
            "message": "Starting...",
            "cycles": 0,
            "done": False,
        }))

        # Run orchestrator with status updates
        await _run_with_websocket_updates(orchestrator, websocket)

        # Final status
        await websocket.send_text(json.dumps({
            "type": "status",
            "message": "Goal complete!",
            "cycles": orchestrator.cycles,
            "done": True,
        }))

    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        try:
            await websocket.send_text(json.dumps({"type": "log", "message": f"Error: {e}"}))
        except RuntimeError:
            pass
    finally:
        _current_orchestrator = None


async def _run_with_websocket_updates(orchestrator: Orchestrator, websocket: WebSocket) -> None:
    """Run orchestrator and send updates via websocket."""
    # We need to monkey-patch the console.print to send updates
    original_cycle = orchestrator._cycle

    async def cycle_with_updates():
        # Send cycle start
        await websocket.send_text(json.dumps({
            "type": "status",
            "message": f"Starting cycle {orchestrator.cycles + 1}",
            "cycles": orchestrator.cycles + 1,
            "done": False,
        }))

        await original_cycle()

        # Send task updates
        if orchestrator.plan:
            await websocket.send_text(json.dumps({
                "type": "tasks",
                "tasks": [t.model_dump() for t in orchestrator.plan.tasks],
            }))

    orchestrator._cycle = cycle_with_updates  # type: ignore
    await orchestrator.run()


@app.get("/api/status")
async def get_status():
    """Get current orchestrator status."""
    if _current_orchestrator:
        return _current_orchestrator.get_status()
    return {"status": "idle"}


@app.post("/api/start")
async def start_goal(request: StartRequest):
    """Start a new goal via REST API."""
    global _current_orchestrator
    if _current_orchestrator:
        return {"error": "Orchestrator already running"}

    from furrow.llm import LLMClient
    client = LLMClient()
    if request.model:
        client.settings.model = request.model

    orchestrator = Orchestrator(goal=request.goal, client=client)
    _current_orchestrator = orchestrator

    # Run in background
    asyncio.create_task(orchestrator.run())

    return {"status": "started", "goal": request.goal}


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)