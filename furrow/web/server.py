from __future__ import annotations

import asyncio
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
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
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


class WebSocketStream:
    """Stream stdout/stderr to a WebSocket."""
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.buffer = StringIO()
    
    def write(self, data: str) -> int:
        self.buffer.write(data)
        # Send complete lines
        value = self.buffer.getvalue()
        lines = value.split('\n')
        # Keep the last incomplete line in buffer
        self.buffer = StringIO(lines[-1])
        # Send complete lines
        for line in lines[:-1]:
            if line:
                asyncio.create_task(self.websocket.send_text(line))
        return len(data)
    
    def flush(self) -> None:
        # Send any remaining buffer content
        value = self.buffer.getvalue()
        if value:
            asyncio.create_task(self.websocket.send_text(value))
            self.buffer = StringIO()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    stream = WebSocketStream(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        # Create orchestrator with custom console that writes to websocket
        from rich.console import Console
        ws_console = Console(file=stream, force_terminal=False, force_jupyter=False)
        
        # Monkey-patch the console in orchestrator module
        import furrow.core.orchestrator as orchestrator_module
        original_console = orchestrator_module.console
        orchestrator_module.console = ws_console
        
        try:
            orchestrator = Orchestrator(goal=goal)
            await orchestrator.run()
        finally:
            orchestrator_module.console = original_console
            stream.flush()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"Error: {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
