from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class WebSocketConsole(Console):
    def __init__(self, websocket: WebSocket, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.websocket = websocket
        self._buffer = StringIO()

    def print(self, *args, **kwargs):
        # Capture output to buffer
        with redirect_stdout(self._buffer), redirect_stderr(self._buffer):
            super().print(*args, **kwargs)
        # Send buffered content to websocket
        content = self._buffer.getvalue()
        if content:
            asyncio.create_task(self._send(content))
            self._buffer.truncate(0)
            self._buffer.seek(0)

    async def _send(self, content: str):
        try:
            await self.websocket.send_text(content)
        except Exception:
            pass


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 1rem; }
  h1 { color: #4CAF50; }
  #form { margin-bottom: 1rem; }
  #goal { width: 70%; padding: 0.5rem; font-size: 1rem; }
  button { padding: 0.5rem 1rem; font-size: 1rem; background: #4CAF50; color: white; border: none; cursor: pointer; }
  button:disabled { background: #ccc; cursor: not-allowed; }
  #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; height: 500px; overflow-y: auto; white-space: pre-wrap; font-family: 'Monaco', 'Menlo', monospace; }
  .cycle { color: #4fc3f7; font-weight: bold; }
  .success { color: #81c784; }
  .error { color: #e57373; }
  .warning { color: #ffb74d; }
  .info { color: #64b5f6; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal (e.g., 'Create a REST API with user CRUD')" required />
    <button type="submit" id="btn">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const btn = document.getElementById('btn');
    let ws = null;
    
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      if (!goal) return;
      
      btn.disabled = true;
      btn.textContent = 'Running...';
      out.textContent = '';
      
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const line = ev.data;
        // Color code based on content
        let cls = '';
        if (line.includes('completed') || line.includes('passed')) cls = 'success';
        else if (line.includes('failed') || line.includes('Error')) cls = 'error';
        else if (line.includes('Cycle')) cls = 'cycle';
        else if (line.includes('Warning') || line.includes('⏳')) cls = 'warning';
        else if (line.includes('Planning') || line.includes('Testing') || line.includes('Executing')) cls = 'info';
        
        const span = document.createElement('span');
        span.className = cls;
        span.textContent = line + '\\n';
        out.appendChild(span);
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => {
        out.textContent += '\\n[Connection closed]\\n';
        btn.disabled = false;
        btn.textContent = 'Start';
      };
      ws.onerror = (err) => {
        out.textContent += '\\n[WebSocket error]\\n';
        btn.disabled = false;
        btn.textContent = 'Start';
      };
      ws.send(JSON.stringify({goal: goal}));
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
        
        # Create a custom console that streams to websocket
        ws_console = WebSocketConsole(websocket, force_terminal=True, color_system="truecolor")
        
        # We need to monkey-patch the console in the orchestrator
        # For now, create orchestrator with custom client that uses ws_console
        from furrow.llm import LLMClient
        
        client = LLMClient()
        
        # Create orchestrator manually to inject our console
        orchestrator = Orchestrator(goal=goal, client=client)
        # Replace the console in the module
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
            await websocket.send_text(f"Error: {e}")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
