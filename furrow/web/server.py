from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console
from rich.status import Status

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


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
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; 
               white-space: pre-wrap; font-size: 14px; line-height: 1.5; max-height: 70vh; overflow: auto; }
        .green { color: #4ec9b0; }
        .red { color: #f44747; }
        .yellow { color: #dcdcaa; }
        .cyan { color: #9cdcfe; }
        .blue { color: #569cd6; }
        input { padding: 0.5rem; font-size: 16px; width: 70%; }
        button { padding: 0.5rem 1rem; font-size: 16px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <form id="form">
        <input id="goal" placeholder="Enter goal (e.g., Add JWT auth to the API)" required />
        <button type="submit">Start</button>
    </form>
    <pre id="out"></pre>
    <script>
        const form = document.getElementById('form');
        const out = document.getElementById('out');
        form.onsubmit = async (e) => {
            e.preventDefault();
            out.textContent += '\\nStarting...\\n';
            const ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                try {
                    const msg = JSON.parse(ev.data);
                    if (msg.type === 'log') {
                        out.textContent += msg.content + '\\n';
                        out.scrollTop = out.scrollHeight;
                    }
                } catch {
                    out.textContent += ev.data + '\\n';
                    out.scrollTop = out.scrollHeight;
                }
            };
            ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
            ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
        };
    </script>
</body>
</html>
""")


class WebSocketConsole(Console):
    """A console-like object that sends output to a WebSocket."""
    
    def __init__(self, websocket: WebSocket) -> None:
        super().__init__(force_terminal=True, width=120)
        self.websocket = websocket
    
    def print(self, *args: Any, **kwargs: Any) -> None:
        # Convert rich markup to plain text for now
        text = " ".join(str(a) for a in args)
        # Strip rich markup tags
        text = re.sub(r'\[/?[a-zA-Z#]+[^\]]*\]', '', text)
        asyncio.create_task(self._send(text))
    
    async def _send(self, text: str) -> None:
        try:
            await self.websocket.send_json({"type": "log", "content": text})
        except Exception:
            pass
    
    def status(self, *args: Any, **kwargs: Any) -> Status:
        """Return a dummy status that doesn't do anything."""
        # We can't really use Rich's Status over WebSocket, so return a no-op
        return Status("", console=self)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        # Create a custom console that sends to websocket
        ws_console = WebSocketConsole(websocket)
        
        # Monkey-patch console in orchestrator
        import furrow.core.orchestrator as orchestrator_module
        original_console = orchestrator_module.console
        orchestrator_module.console = ws_console
        
        try:
            orchestrator = Orchestrator(goal=goal)
            await orchestrator.run()
        finally:
            orchestrator_module.console = original_console
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "log", "content": f"Error: {e}"})
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
