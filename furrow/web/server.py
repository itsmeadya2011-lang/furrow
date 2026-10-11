from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class OrchestratorState(BaseModel):
    goal: str
    cycles: int
    tasks: list[dict]
    is_running: bool


# Store active orchestrator for web interface
_active_orchestrator: Orchestrator | None = None
_websocket: WebSocket | None = None


class WebSocketConsole:
    """Redirect console output to websocket."""
    
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
    
    def print(self, *args, **kwargs) -> None:
        # Send to websocket
        message = " ".join(str(arg) for arg in args)
        asyncio.create_task(self.websocket.send_text(message))


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; }
        #out { background: #1e1e1e; color: #d4d4d4; padding: 15px; height: 500px; overflow-y: auto; border-radius: 4px; }
        .error { color: #f44747; }
        .success { color: #4ec9b0; }
        .warning { color: #dcdcaa; }
        .info { color: #9cdcfe; }
        form { margin-bottom: 20px; }
        input[type="text"] { width: 70%; padding: 8px; font-family: monospace; }
        button { padding: 8px 16px; cursor: pointer; }
        #status { margin-bottom: 10px; font-weight: bold; }
        .running { color: #4ec9b0; }
        .stopped { color: #f44747; }
    </style>
</head>
<body>
    <h1>Furrow Web Interface</h1>
    <div id="status" class="stopped">Status: Stopped</div>
    <form id="form">
        <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
        <button type="submit" id="startBtn">Start</button>
        <button type="button" id="stopBtn" disabled>Stop</button>
    </form>
    <pre id="out"></pre>
    <script>
        const form = document.getElementById('form');
        const out = document.getElementById('out');
        const status = document.getElementById('status');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        let ws = null;
        
        function appendOutput(text, className = '') {
            const line = document.createElement('div');
            line.textContent = text;
            if (className) line.className = className;
            out.appendChild(line);
            out.scrollTop = out.scrollHeight;
        }
        
        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value;
            if (!goal) return;
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            status.textContent = 'Status: Running...';
            status.className = 'running';
            appendOutput('\\nStarting Furrow...\\n', 'info');
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                const text = ev.data;
                if (text.includes('failed') || text.includes('Error') || text.includes('error:')) {
                    appendOutput(text, 'error');
                } else if (text.includes('passed') || text.includes('completed') || text.includes('Goal complete')) {
                    appendOutput(text, 'success');
                } else if (text.includes('Warning') || text.includes('warning')) {
                    appendOutput(text, 'warning');
                } else {
                    appendOutput(text);
                }
            };
            ws.onclose = () => {
                appendOutput('\\nConnection closed.\\n', 'warning');
                startBtn.disabled = false;
                stopBtn.disabled = true;
                status.textContent = 'Status: Stopped';
                status.className = 'stopped';
            };
            ws.onerror = (err) => {
                appendOutput('\\nWebSocket error: ' + err.message, 'error');
            };
            
            // Wait for connection then send goal
            ws.onopen = () => {
                ws.send(JSON.stringify({goal: goal}));
            };
        };
        
        stopBtn.onclick = () => {
            if (ws) {
                ws.close();
            }
        };
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    global _active_orchestrator, _websocket
    await websocket.accept()
    _websocket = websocket
    
    # Create a custom console that sends to websocket
    from io import StringIO

    from rich.console import Console
    
    # We'll capture rich output and send via websocket
    string_io = StringIO()
    ws_console = Console(file=string_io, force_terminal=True, width=100)
    
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        
        if model:
            from furrow.config import settings
            settings.model = model
        
        _active_orchestrator = Orchestrator(goal=goal)
        
        # Monkey-patch console in orchestrator
        import furrow.core.orchestrator as orch_module
        original_console = orch_module.console
        orch_module.console = ws_console
        
        try:
            await _active_orchestrator.run()
        finally:
            orch_module.console = original_console
            
    except WebSocketDisconnect:
        pass
    except (RuntimeError, ValueError, OSError) as e:
        await websocket.send_text(f"Error: {e}")
    finally:
        _websocket = None
        _active_orchestrator = None


@app.get("/api/state")
async def get_state() -> JSONResponse:
    if _active_orchestrator is None:
        raise HTTPException(status_code=404, detail="No active orchestrator")
    
    return JSONResponse({
        "goal": _active_orchestrator.goal,
        "cycles": _active_orchestrator.cycles,
        "tasks": [t.model_dump() for t in _active_orchestrator.all_tasks],
        "is_running": True,
    })


@app.post("/api/stop")
async def stop_orchestrator() -> JSONResponse:
    global _active_orchestrator, _websocket
    if _active_orchestrator:
        _active_orchestrator._save_state()
        _active_orchestrator = None
    if _websocket:
        await _websocket.close()
        _websocket = None
    return JSONResponse({"status": "stopped"})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
