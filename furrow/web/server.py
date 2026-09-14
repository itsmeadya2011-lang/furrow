from __future__ import annotations

import asyncio
from io import StringIO
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketBroadcaster(Console):
    """A rich Console subclass that buffers output and sends it to a WebSocket."""

    def __init__(self, websocket: WebSocket) -> None:
        super().__init__(file=StringIO(), force_terminal=False, width=120)
        self.websocket = websocket
        self.queue: asyncio.Queue = asyncio.Queue()

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Capture rich output to a string and queue it for sending."""
        self.file = StringIO()
        super().print(*args, **kwargs)
        text = self.file.getvalue()
        self.queue.put_nowait(text)

    async def drain(self) -> None:
        """Send all queued messages to the WebSocket."""
        while not self.queue.empty():
            try:
                message = self.queue.get_nowait()
                await self.websocket.send_text(message)
            except asyncio.QueueEmpty:
                break

    async def send_raw(self, message: str) -> None:
        """Send a raw message directly to the WebSocket."""
        await self.websocket.send_text(message)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    * { box-sizing: border-box; }
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #1a1b26; 
      color: #c0caf5; 
      margin: 0; 
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: #16161e;
      border-bottom: 1px solid #292e42;
      padding: 1rem 2rem;
    }
    header h1 { 
      margin: 0; 
      font-size: 1.5rem; 
      font-weight: 600;
      color: #7aa2f7;
    }
    main { 
      flex: 1; 
      padding: 2rem; 
      display: flex; 
      flex-direction: column; 
      gap: 1rem;
      max-width: 1200px;
      margin: 0 auto;
      width: 100%;
    }
    .input-group {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    #goal {
      flex: 1;
      min-width: 300px;
      padding: 0.75rem 1rem;
      background: #24283b;
      border: 1px solid #292e42;
      border-radius: 6px;
      color: #c0caf5;
      font-size: 1rem;
      outline: none;
    }
    #goal:focus { border-color: #7aa2f7; }
    #goal::placeholder { color: #565f89; }
    button {
      padding: 0.75rem 1.5rem;
      background: #7aa2f7;
      border: none;
      border-radius: 6px;
      color: #1a1b26;
      font-weight: 600;
      font-size: 1rem;
      cursor: pointer;
      transition: background 0.2s;
    }
    button:hover { background: #5d8de6; }
    button:disabled { 
      background: #414868; 
      color: #565f89; 
      cursor: not-allowed; 
    }
    #out {
      flex: 1;
      background: #16161e;
      border: 1px solid #292e42;
      border-radius: 8px;
      padding: 1rem;
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 0.875rem;
      line-height: 1.6;
      overflow-y: auto;
      white-space: pre-wrap;
      word-wrap: break-word;
      min-height: 400px;
    }
    .log-line { margin: 0.25rem 0; }
    .log-info { color: #7aa2f7; }
    .log-success { color: #9ece6a; }
    .log-warning { color: #e0af68; }
    .log-error { color: #f7768e; }
    .log-dim { color: #565f89; }
    .status-badge {
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
    }
    .status-running { background: #2e3c64; color: #7aa2f7; }
    .status-done { background: #27382e; color: #9ece6a; }
    .status-error { background: #3d2a32; color: #f7768e; }
    footer {
      padding: 1rem 2rem;
      background: #16161e;
      border-top: 1px solid #292e42;
      text-align: center;
      color: #565f89;
      font-size: 0.875rem;
    }
  </style>
</head>
<body>
  <header>
    <h1>Furrow</h1>
  </header>
  <main>
    <div class="input-group">
      <input id="goal" placeholder="Enter goal (e.g., 'Create a REST API for a todo app')" required />
      <button id="startBtn">Start</button>
    </div>
    <div id="status" class="status-badge status-dim" style="display:none;"></div>
    <pre id="out"></pre>
  </main>
  <footer>WebSocket-connected orchestrator • Real-time output</footer>
  <script>
    const goalInput = document.getElementById('goal');
    const startBtn = document.getElementById('startBtn');
    const statusEl = document.getElementById('status');
    const out = document.getElementById('out');
    let ws = null;

    function setStatus(text, className) {
      statusEl.textContent = text;
      statusEl.className = 'status-badge ' + className;
      statusEl.style.display = 'inline-block';
    }

    function appendLog(text) {
      const line = document.createElement('div');
      line.className = 'log-line';
      if (text.includes('[green]') || text.includes('completed') || text.includes('passed')) {
        line.classList.add('log-success');
      } else if (text.includes('[red]') || text.includes('failed') || text.includes('error')) {
        line.classList.add('log-error');
      } else if (text.includes('[yellow]') || text.includes('warning') || text.includes('Will attempt')) {
        line.classList.add('log-warning');
      } else if (text.includes('[cyan]') || text.includes('Cycle')) {
        line.classList.add('log-info');
      } else if (text.includes('[dim]') || text.includes('═══')) {
        line.classList.add('log-dim');
      }
      line.textContent = text.replace(/\[[\w\/]+\]/g, '');
      out.appendChild(line);
      out.scrollTop = out.scrollHeight;
    }

    startBtn.onclick = async () => {
      const goal = goalInput.value.trim();
      if (!goal) return;
      
      out.textContent = '';
      startBtn.disabled = true;
      goalInput.disabled = true;
      setStatus('Connecting...', 'status-running');
      
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => {
        setStatus('Running', 'status-running');
        ws.send(JSON.stringify({goal}));
      };
      ws.onmessage = (ev) => {
        const data = ev.data;
        if (data === 'Done.' || data.trim() === 'Goal complete. Halting.') {
          setStatus('Done', 'status-done');
          startBtn.disabled = false;
          goalInput.disabled = false;
          if (ws) ws.close();
        } else {
          appendLog(data);
        }
      };
      ws.onclose = () => {
        if (startBtn.disabled) {
          setStatus('Disconnected', 'status-error');
          startBtn.disabled = false;
          goalInput.disabled = false;
        }
      };
      ws.onerror = () => {
        setStatus('Error', 'status-error');
        startBtn.disabled = false;
        goalInput.disabled = false;
      };
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    broadcaster = WebSocketBroadcaster(websocket)

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        orchestrator = Orchestrator(goal=goal, console=broadcaster)
        run_task = asyncio.create_task(orchestrator.run())

        while not run_task.done():
            await broadcaster.drain()
            await asyncio.sleep(0.05)

        await broadcaster.drain()
        await broadcaster.send_raw("Done.")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await broadcaster.send_raw(f"[red]Error: {e}[/red]")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)