from __future__ import annotations

import asyncio
import sys
from contextlib import redirect_stderr, redirect_stdout
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
<head><title>Furrow</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; 
         min-height: 400px; overflow-y: auto; white-space: pre-wrap; }
  input { width: 70%; padding: 0.5rem; }
  button { padding: 0.5rem 1rem; }
  .green { color: #4ec9b0; }
  .red { color: #f44747; }
  .yellow { color: #dcdcaa; }
  .cyan { color: #9cdcfe; }
</style>
</head>
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
      ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


class OutputCapture:
    """Capture stdout/stderr and stream to websocket."""
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.stdout_buffer = StringIO()
        self.stderr_buffer = StringIO()
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    def write(self, text: str) -> None:
        if text.strip():
            asyncio.create_task(self.websocket.send_text(text.rstrip()))
        self._original_stdout.write(text)

    def flush(self) -> None:
        self._original_stdout.flush()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        # Capture console output
        capture = OutputCapture(websocket)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = capture
        sys.stderr = capture
        
        try:
            orchestrator = Orchestrator(goal=goal)
            await orchestrator.run()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"Error: {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)