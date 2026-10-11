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


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketConsole:
    """Console that streams output to a WebSocket."""
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.buffer = ""
    
    def print(self, *args, **kwargs) -> None:
        # Capture rich output
        import io
        from rich.console import Console
        from rich.text import Text
        
        # Create a console that captures output
        capture_console = Console(file=io.StringIO(), force_terminal=True, width=120)
        capture_console.print(*args, **kwargs)
        output = capture_console.file.getvalue()
        
        # Send to websocket
        asyncio.create_task(self._send(output))
    
    async def _send(self, text: str) -> None:
        try:
            await self.websocket.send_text(text)
        except Exception:
            pass


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
  h1 { color: #4CAF50; }
  form { margin-bottom: 20px; }
  input { padding: 10px; width: 70%; background: #333; border: 1px solid #555; color: #fff; }
  button { padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; }
  button:hover { background: #45a049; }
  #out { background: #0d0d0d; padding: 15px; height: 500px; overflow-y: auto; border: 1px solid #333; white-space: pre-wrap; }
  .green { color: #4CAF50; }
  .red { color: #f44336; }
  .yellow { color: #ffeb3b; }
  .cyan { color: #00bcd4; }
  .blue { color: #2196F3; }
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
      out.textContent += '\\n> Starting: ' + goal + '\\n';
      
      if (ws) ws.close();
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        // Parse ANSI color codes for basic display
        let text = ev.data;
        out.textContent += text;
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
      ws.onopen = () => ws.send(JSON.stringify({goal}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        settings = Settings()
        if data.get("model"):
            settings.model = data["model"]
        
        # Create a custom console that streams to websocket
        ws_console = WebSocketConsole(websocket)
        
        orchestrator = Orchestrator(goal=goal, settings=settings)
        
        # Monkey-patch console for streaming
        import furrow.core.orchestrator as orch_module
        original_console = orch_module.console
        orch_module.console = ws_console
        
        try:
            await orchestrator.run()
        finally:
            orch_module.console = original_console
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"\\n[Error: {e}]\\n")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
