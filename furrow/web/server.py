from __future__ import annotations

import asyncio
import sys
from io import StringIO
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from rich.console import Console

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
        body { font-family: monospace; max-width: 1000px; margin: 0 auto; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
        h1 { color: #4ec9b0; }
        form { margin-bottom: 20px; }
        input { padding: 10px; width: 70%; background: #3c3c3c; border: 1px solid #555; color: #d4d4d4; }
        button { padding: 10px 20px; background: #0e639c; color: white; border: none; cursor: pointer; }
        button:hover { background: #1177bb; }
        #out { background: #1e1e1e; padding: 15px; border: 1px solid #333; white-space: pre-wrap; word-wrap: break-word; }
        .cycle { color: #569cd6; font-weight: bold; }
        .success { color: #4ec9b0; }
        .error { color: #f44747; }
        .warning { color: #dcdcaa; }
        .info { color: #9cdcfe; }
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
    let ws = null;
    
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      out.textContent += '\\n[Starting Furrow...]\\n';
      
      if (ws) {
        ws.close();
      }
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const msg = ev.data;
        out.textContent += msg + '\\n';
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
      ws.onerror = (err) => out.textContent += '\\n[Error: ' + err + ']\\n';
      ws.send(JSON.stringify({goal: goal}));
    };
  </script>
</body>
</html>
""")


class WebSocketConsole:
    """Console that writes to a WebSocket."""
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.buffer = StringIO()
    
    def write(self, text: str) -> None:
        if text.strip():
            # Send each line separately for better streaming
            for line in text.splitlines(keepends=True):
                if line.strip():
                    asyncio.create_task(self.websocket.send_text(line.rstrip()))
    
    def flush(self) -> None:
        pass
    
    # Rich Console compatibility methods
    def print(self, *args, **kwargs) -> None:
        # Convert args to string
        text = " ".join(str(a) for a in args)
        self.write(text)
    
    def __getattr__(self, name):
        # Delegate other methods to a dummy console
        return getattr(Console(), name)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        # Create a custom console that writes to websocket
        ws_console = WebSocketConsole(websocket)
        
        # Run orchestrator with websocket output
        orchestrator = Orchestrator(goal=goal, console=ws_console)
        
        await orchestrator.run()
             
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"[Error: {e}]")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
