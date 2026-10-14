from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class WebSocketConsole:
    """Console that captures output and sends to WebSocket."""
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task = asyncio.create_task(self._sender())

    async def _sender(self):
        while True:
            text = await self._queue.get()
            if text is None:
                break
            try:
                await self.websocket.send_text(text)
            except RuntimeError:
                break

    def print(self, *args, **kwargs):
        # Capture print output
        output = " ".join(str(arg) for arg in args)
        if output:
            self._queue.put_nowait(output)

    async def close(self):
        self._queue.put_nowait(None)
        await self._task


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        h1 { color: #333; }
        #form { margin-bottom: 1rem; }
        #goal { width: 70%; padding: 0.5rem; font-size: 1rem; }
        button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
        #out { 
            background: #1e1e1e; color: #d4d4d4; padding: 1rem; 
            border-radius: 4px; height: 60vh; overflow-y: auto;
            white-space: pre-wrap; font-family: monospace; font-size: 0.9rem;
        }
        .cycle { color: #569cd6; font-weight: bold; }
        .success { color: #4ec9b0; }
        .error { color: #f14c4c; }
        .warning { color: #dcdcaa; }
        .info { color: #9cdcfe; }
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
      const goal = document.getElementById('goal').value;
      out.textContent += '\\nStarting Furrow...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const line = ev.data;
        // Color code output
        let cls = '';
        if (line.includes('═══ Cycle')) cls = 'cycle';
        else if (line.includes('completed') || line.includes('passed')) cls = 'success';
        else if (line.includes('failed') || line.includes('Error')) cls = 'error';
        else if (line.includes('warning') || line.includes('Will attempt')) cls = 'warning';
        else if (line.includes('Planning') || line.includes('Executing') || line.includes('Testing') || line.includes('Linting')) cls = 'info';
        
        if (cls) {
          out.innerHTML += '<span class="' + cls + '">' + line.replace(/</g, '<').replace(/>/g, '>') + '</span>\\n';
        } else {
          out.textContent += line + '\\n';
        }
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\\n--- Connection closed ---\\n';
      ws.onerror = (err) => out.textContent += '\\n--- WebSocket error ---\\n';
      ws.send(JSON.stringify({goal: goal}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    console = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        # Create a custom console that sends to websocket
        console = WebSocketConsole(websocket)
        
        # Monkey-patch the orchestrator's console
        import furrow.core.orchestrator as orch_module
        original_console = orch_module.console
        orch_module.console = console
        
        try:
            orchestrator = Orchestrator(goal=goal)
            await orchestrator.run()
            await websocket.send_text("\n--- Furrow completed ---")
        finally:
            orch_module.console = original_console
            
    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        await websocket.send_text(f"\n--- Error: {e} ---")
    finally:
        if console:
            await console.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
