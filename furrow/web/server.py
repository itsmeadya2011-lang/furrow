from __future__ import annotations

import asyncio
import json
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
from furrow.logging import configure_logging

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketConsole:
    """Capture console output and send to WebSocket."""
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.buffer = StringIO()
    
    def write(self, text: str) -> int:
        self.buffer.write(text)
        # Send complete lines
        if '\n' in text:
            lines = self.buffer.getvalue().split('\n')
            # Keep the last partial line in buffer
            self.buffer = StringIO(lines[-1])
            # Send complete lines
            for line in lines[:-1]:
                if line.strip():
                    asyncio.create_task(self.websocket.send_text(line))
        return len(text)
    
    def flush(self) -> None:
        pass
    
    async def send_remaining(self) -> None:
        remaining = self.buffer.getvalue()
        if remaining.strip():
            await self.websocket.send_text(remaining)


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
    input { padding: 10px; width: 60%; background: #3c3c3c; border: 1px solid #555; color: #d4d4d4; }
    button { padding: 10px 20px; background: #4ec9b0; border: none; color: #1e1e1e; cursor: pointer; font-weight: bold; }
    button:hover { background: #3db8a0; }
    #out { background: #252526; padding: 15px; height: 500px; overflow-y: auto; border: 1px solid #3c3c3c; white-space: pre-wrap; }
    .cycle { color: #9cdcfe; font-weight: bold; }
    .success { color: #4ec9b0; }
    .error { color: #f44747; }
    .warning { color: #dcdcaa; }
    .info { color: #9cdcfe; }
  </style>
</head>
<body>
  <h1>Furrow Web Interface</h1>
  <form id="form">
    <input id="goal" placeholder="Enter your goal (e.g., 'Add JWT auth to the API')" required />
    <button type="submit">Start Furrow</button>
  </form>
  <div id="out"></div>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    let ws = null;
    
    function addLine(text, className = '') {
      const div = document.createElement('div');
      div.className = className;
      div.textContent = text;
      out.appendChild(div);
      out.scrollTop = out.scrollHeight;
    }
    
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      if (!goal) return;
      
      addLine('Starting Furrow...', 'info');
      document.getElementById('goal').value = '';
      
      if (ws) {
        ws.close();
      }
      
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        if (data.type === 'output') {
          addLine(data.text, data.class || '');
        } else if (data.type === 'status') {
          addLine(data.text, 'info');
        }
      };
      ws.onclose = () => addLine('\\nConnection closed.', 'warning');
      ws.onerror = () => addLine('\\nConnection error.', 'error');
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
    await websocket.accept()
    ws_console = WebSocketConsole(websocket)
    
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        if not goal:
            await websocket.send_text(json.dumps({"type": "output", "text": "Error: No goal provided", "class": "error"}))
            return
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = ws_console
        sys.stderr = ws_console
        
        try:
            orchestrator = Orchestrator(goal=goal)
            await orchestrator.run()
            await ws_console.send_remaining()
            await websocket.send_text(json.dumps({"type": "output", "text": "\n✓ Furrow completed successfully!", "class": "success"}))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({"type": "output", "text": f"\n✗ Error: {e}", "class": "error"}))


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    configure_logging()
    uvicorn.run(app, host=host, port=port)
