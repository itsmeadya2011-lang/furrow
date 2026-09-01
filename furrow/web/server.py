from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Track active WebSocket connections for streaming
_active_connections: list[WebSocket] = []


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Furrow</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334159;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --green: #4ade80;
      --cyan: #06b6d4;
      --yellow: #fbbf24;
      --red: #f87171;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Fira Code', 'Monaco', 'Consolas', monospace;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 2rem;
    }
    .container { max-width: 900px; margin: 0 auto; }
    h1 {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 1rem;
      font-size: 1.75rem;
    }
    h1 .dot {
      width: 12px; height: 12px;
      border-radius: 50%;
      background: var(--green);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.5rem;
      margin-bottom: 1rem;
    }
    .input-group {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1rem;
    }
    input[type="text"] {
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.5rem 0.75rem;
      border-radius: 4px;
      font-size: 0.9rem;
    }
    button {
      background: var(--cyan);
      color: #0f172a;
      border: none;
      padding: 0.5rem 1.25rem;
      border-radius: 4px;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.9rem;
    }
    button:hover { background: #0891b2; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    #log {
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 1rem;
      height: 400px;
      overflow-y: auto;
      font-size: 0.8rem;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .status-stopped { color: var(--yellow); }
  </style>
</head>
<body>
  <div class="container">
    <h1>
      <span class="dot"></span>
      Furrow — Autonomous Development Loop
    </h1>
    <div class="card">
      <div class="input-group">
        <input id="goal" type="text" placeholder="Enter your development goal..." autocomplete="off" />
        <button id="startBtn" type="submit">Start</button>
      </div>
      <div id="log"></div>
    </div>
  </div>
  <script>
    const goalInput = document.getElementById('goal');
    const startBtn = document.getElementById('startBtn');
    const logEl = document.getElementById('log');
    const form = document.createElement('form');
    form.id = 'form';
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = goalInput.value.trim();
      if (!goal) return;
      startBtn.disabled = true;
      logEl.textContent = '';
      logEl.textContent += '\\n\\x1b[33mStarting Furrow...\\x1b[0m\\n';

      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        logEl.textContent += ev.data + '\\n';
        logEl.scrollTop = logEl.scrollHeight;
      };
      ws.onopen = () => {
        ws.send(JSON.stringify({goal: goal}));
      };
      ws.onerror = (ev) => {
        logEl.textContent += '\\n\\x1b[31mWebSocket error. Is the server running?\\x1b[0m\\n';
      };
      ws.onclose = () => {
        startBtn.disabled = false;
        logEl.textContent += '\\n\\x1b[33mConnection closed.\\x1b[0m\\n';
      };
    };
    form.appendChild(goalInput);
    form.appendChild(startBtn);
    document.querySelector('.input-group').appendChild(form);
    // Re-attach onsubmit since form replaced children
    form.onsubmit = (e) => {
      e.preventDefault();
      const goal = goalInput.value.trim();
      if (!goal) return;
      startBtn.disabled = true;
      logEl.textContent = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        logEl.textContent += ev.data + '\\n';
        logEl.scrollTop = logEl.scrollHeight;
      };
      ws.onopen = () => ws.send(JSON.stringify({goal: goal}));
      ws.onerror = () => { logEl.textContent += '\\n[error] WebSocket error.\\n'; };
      ws.onclose = () => { startBtn.disabled = false; };
    };
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=HTML_TEMPLATE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _active_connections.append(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal:
            await websocket.send_json({"type": "error", "message": "No goal provided."})
            return

        orchestrator = Orchestrator(goal=goal)

        # Stream orchestrator output by capturing rich console output
        from io import StringIO
        from rich.console import Console

        buffer = StringIO()
        console = Console(file=buffer, width=120, force_terminal=False, color_system=None)

        # Patch the orchestrator's console for streaming
        import furrow.core.orchestrator as orch_module
        original_console = orch_module.console
        orch_module.console = console

        try:
            # Run in a background task while relaying output
            async def _run_and_stream():
                try:
                    await orchestrator.run()
                    await websocket.send_text("[done] Goal complete.")
                except Exception as e:
                    await websocket.send_text(f"[error] {e}")

            task = asyncio.create_task(_run_and_stream())
            while not task.done():
                await asyncio.sleep(0.2)
                output = buffer.getvalue()
                buffer.seek(0)
                buffer.truncate()
                if output.strip():
                    await websocket.send_text(output.rstrip())
            # Final flush
            remaining = buffer.getvalue()
            if remaining.strip():
                await websocket.send_text(remaining.rstrip())
        finally:
            orch_module.console = original_console
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _active_connections:
            _active_connections.remove(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
