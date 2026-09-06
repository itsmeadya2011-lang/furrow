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


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #0d1117;
      color: #c9d1d9;
      padding: 2rem;
      line-height: 1.6;
    }
    .container { max-width: 900px; margin: 0 auto; }
    h1 { color: #58a6ff; margin-bottom: 1.5rem; }
    .form { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; }
    input[type="text"] {
      flex: 1;
      padding: 0.75rem 1rem;
      border-radius: 6px;
      border: 1px solid #30363d;
      background: #161b22;
      color: #c9d1d9;
      font-size: 1rem;
    }
    button {
      padding: 0.75rem 1.5rem;
      border-radius: 6px;
      border: none;
      background: #238636;
      color: #fff;
      font-size: 1rem;
      cursor: pointer;
    }
    button:hover { background: #2ea043; }
    button:disabled { background: #30363d; cursor: not-allowed; }
    .log {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 1rem;
      font-family: 'SFMono-Regular', Consolas, monospace;
      font-size: 0.875rem;
      white-space: pre-wrap;
      max-height: 70vh;
      overflow-y: auto;
    }
    .cycle { color: #58a6ff; font-weight: bold; }
    .success { color: #3fb950; }
    .error { color: #f85149; }
    .warn { color: #d29922; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Furrow</h1>
    <form id="form" class="form">
      <input id="goal" placeholder="Enter your goal" required />
      <button type="submit" id="startBtn">Start</button>
    </form>
    <div id="out" class="log"></div>
  </div>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const startBtn = document.getElementById('startBtn');
    function append(text, className) {
      const span = document.createElement('span');
      span.className = className || '';
      span.textContent = text + '\\n';
      out.appendChild(span);
      out.scrollTop = out.scrollHeight;
    }
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      startBtn.disabled = true;
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'log') append(msg.text);
          else if (msg.type === 'cycle') append('═══ Cycle ' + msg.cycle + ' ═══', 'cycle');
          else if (msg.type === 'success') append(msg.text, 'success');
          else if (msg.type === 'error') append(msg.text, 'error');
          else if (msg.type === 'warn') append(msg.text, 'warn');
          else append(ev.data);
        } catch {
          append(ev.data);
        }
      };
      ws.onclose = () => {
        append('\\nSession closed.', 'warn');
        startBtn.disabled = false;
      };
      ws.onerror = () => {
        append('\\nConnection error.', 'error');
        startBtn.disabled = false;
      };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    def send(event: str, data: dict) -> None:
        asyncio.create_task(websocket.send_json({"type": event, **data}))

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)

        send("log", {"text": f"Furrow starting. Goal: {goal}"})

        max_cycles = orchestrator.client.settings.max_cycles
        for cycle in range(1, 1000):
            send("cycle", {"cycle": cycle})
            await orchestrator._cycle()
            if orchestrator._is_done():
                send("success", {"text": "Goal complete. Halting."})
                break
            if max_cycles > 0 and cycle >= max_cycles:
                send("warn", {"text": f"Reached max_cycles ({max_cycles}). Halting."})
                break
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
