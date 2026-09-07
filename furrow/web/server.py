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
<head><title>Furrow</title>
<style>
  body { font-family: monospace; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
  h1 { color: #4CAF50; }
  #form { margin-bottom: 20px; }
  input { padding: 8px; width: 300px; background: #2d2d2d; border: 1px solid #444; color: #d4d4d4; }
  button { padding: 8px 16px; background: #4CAF50; color: white; border: none; cursor: pointer; }
  button.stop { background: #f44336; }
  #out { background: #2d2d2d; padding: 10px; border-radius: 4px; white-space: pre-wrap; max-height: 80vh; overflow-y: auto; }
  .status { color: #2196F3; }
  .success { color: #4CAF50; }
  .error { color: #f44336; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <div id="status"></div>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const status = document.getElementById('status');
    let ws = null;
    let running = false;

    form.onsubmit = async (e) => {
      e.preventDefault();
      if (running) return;
      running = true;
      out.textContent = '';
      status.innerHTML = '<span class="status">Starting...</span>';
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'log') {
          out.textContent += msg.data + '\\n';
        } else if (msg.type === 'status') {
          status.innerHTML = '<span class="status">' + msg.data + '</span>';
        } else if (msg.type === 'done') {
          status.innerHTML = '<span class="success">' + msg.data + '</span>';
          running = false;
        } else if (msg.type === 'error') {
          status.innerHTML = '<span class="error">' + msg.data + '</span>';
          running = false;
        }
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => {
        if (running) status.innerHTML = '<span class="error">Disconnected</span>';
        running = false;
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
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        async def _run_orchestrator() -> None:
            try:
                orchestrator = Orchestrator(goal=goal)
                if orchestrator.load_state():
                    await websocket.send_json({"type": "log", "data": "[yellow]Resumed from previous state.[/yellow]"})
                # Monkey-patch console to stream logs
                from furrow.core import orchestrator as orch_mod
                original_print = orch_mod.console.print
                def _stream_print(*args, **kwargs):
                    original_print(*args, **kwargs)
                    text = " ".join(str(a) for a in args)
                    import asyncio as _asyncio
                    _asyncio.ensure_future(websocket.send_json({"type": "log", "data": text}))
                orch_mod.console.print = _stream_print
                await orchestrator.run()
                await websocket.send_json({"type": "done", "data": "Goal complete."})
            except Exception as e:
                await websocket.send_json({"type": "error", "data": str(e)})

        await _run_orchestrator()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
