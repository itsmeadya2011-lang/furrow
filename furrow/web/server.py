from __future__ import annotations

import asyncio
import sys
from contextlib import redirect_stdout, redirect_stderr
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
    """Redirects stdout/stderr to a WebSocket."""
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.buffer = StringIO()

    def write(self, data: str) -> int:
        self.buffer.write(data)
        # Flush on newlines
        if '\n' in data:
            self.flush()
        return len(data)

    def flush(self) -> None:
        content = self.buffer.getvalue()
        if content:
            try:
                asyncio.create_task(self.websocket.send_text(content))
            except RuntimeError:
                pass  # Event loop not running
            self.buffer.truncate(0)
            self.buffer.seek(0)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        # Create a custom console that writes to WebSocket
        stream = WebSocketStream(websocket)
        
        # Run orchestrator with output redirected
        orchestrator = Orchestrator(goal=goal)
        
        # We need to capture rich console output
        from rich.console import Console
        ws_console = Console(file=stream, force_terminal=False)
        
        # Monkey-patch the orchestrator's console
        import furrow.core.orchestrator as orch_module
        original_console = orch_module.console
        orch_module.console = ws_console
        
        try:
            await orchestrator.run()
        finally:
            orch_module.console = original_console
            stream.flush()
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
