from __future__ import annotations

import asyncio
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Global state for web UI
_active_websockets: list[WebSocket] = []
_orchestrator: Orchestrator | None = None


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


async def broadcast(message: str) -> None:
    """Broadcast message to all connected websockets."""
    for ws in _active_websockets[:]:
        try:
            await ws.send_text(message)
        except Exception:
            _active_websockets.remove(ws)


class WebConsole:
    """Console that broadcasts to websockets instead of stdout."""
    
    def print(self, *args, **kwargs) -> None:
        message = " ".join(str(a) for a in args)
        asyncio.create_task(broadcast(message))
    
    def log(self, *args, **kwargs) -> None:
        self.print(*args, **kwargs)


web_console = WebConsole()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; margin: 20px; background: #1e1e1e; color: #d4d4d4; }
        h1 { color: #4ec9b0; }
        #form { margin-bottom: 20px; }
        input { padding: 8px; font-size: 16px; width: 60%; background: #3c3c3c; border: 1px solid #555; color: #d4d4d4; }
        button { padding: 8px 16px; font-size: 16px; background: #0e639c; color: white; border: none; cursor: pointer; }
        button:hover { background: #1177bb; }
        button:disabled { background: #555; cursor: not-allowed; }
        #output { 
            background: #1e1e1e; 
            border: 1px solid #3c3c3c; 
            padding: 15px; 
            height: 500px; 
            overflow-y: auto; 
            white-space: pre-wrap; 
            font-family: 'Fira Code', monospace;
            font-size: 13px;
            line-height: 1.5;
        }
        .cycle { color: #9cdcfe; font-weight: bold; margin-top: 10px; }
        .task-done { color: #4ec9b0; }
        .task-fail { color: #f44747; }
        .plan { color: #ce9178; }
        .test-pass { color: #4ec9b0; }
        .test-fail { color: #f44747; }
        .summary { color: #dcdcaa; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <form id="form">
        <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
        <button type="submit" id="startBtn">Start</button>
    </form>
    <pre id="output"></pre>
    <script>
        const form = document.getElementById('form');
        const goalInput = document.getElementById('goal');
        const startBtn = document.getElementById('startBtn');
        const output = document.getElementById('output');
        let ws = null;
        
        function append(msg) {
            output.textContent += msg + '\\n';
            output.scrollTop = output.scrollHeight;
        }
        
        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = goalInput.value.trim();
            if (!goal) return;
            
            startBtn.disabled = true;
            startBtn.textContent = 'Running...';
            append('\\n[Starting Furrow...]\\n');
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => append(ev.data);
            ws.onclose = () => {
                append('\\n[Connection closed]');
                startBtn.disabled = false;
                startBtn.textContent = 'Start';
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
    global _orchestrator
    await websocket.accept()
    _active_websockets.append(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        # Create a custom console that broadcasts to websocket
        _orchestrator = Orchestrator(goal=goal)
        
        # Monkey-patch the console for this orchestrator run
        import furrow.core.orchestrator as orchestrator_module
        original_console = orchestrator_module.console
        orchestrator_module.console = web_console
        
        try:
            await _orchestrator.run()
        finally:
            orchestrator_module.console = original_console
            _orchestrator = None
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _active_websockets:
            _active_websockets.remove(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)