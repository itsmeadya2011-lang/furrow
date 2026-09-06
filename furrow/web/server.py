from __future__ import annotations

import asyncio
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


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; margin: 20px; background: #1a1a2e; color: #e0e0e0; }
  h1 { color: #4CAF50; }
  #form { margin-bottom: 10px; }
  #goal { width: 80%; padding: 8px; background: #16213e; color: #e0e0e0; border: 1px solid #0f3460; }
  button { padding: 8px 16px; background: #4CAF50; color: white; border: none; cursor: pointer; }
  button:hover { background: #45a049; }
  #out { white-space: pre-wrap; word-break: break-word; background: #16213e; padding: 15px; border-radius: 5px; max-height: 70vh; overflow-y: auto; }
  .error { color: #ff6b6b; }
  .success { color: #51cf66; }
  .warning { color: #ffd43b; }
  .info { color: #74c0fc; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <div id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button id="startBtn" type="button">Start</button>
    <button id="stopBtn" type="button" style="display:none;">Stop</button>
  </div>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    let ws = null;

    function addOutput(msg, cls) {
      const div = document.createElement('div');
      if (cls) div.className = cls;
      div.textContent = msg;
      out.appendChild(div);
      out.scrollTop = out.scrollHeight;
    }

    startBtn.onclick = async () => {
      const goal = document.getElementById('goal').value;
      if (!goal) return;
      out.innerHTML = '';
      startBtn.disabled = true;
      stopBtn.style.display = 'inline-block';
      addOutput('Starting...', 'info');
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => addOutput(ev.data);
      ws.onclose = () => { addOutput('\\nClosed.', 'warning'); startBtn.disabled = false; stopBtn.style.display = 'none'; };
      ws.onopen = () => ws.send(JSON.stringify({goal: goal}));
    };

    stopBtn.onclick = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({action: 'stop'}));
        addOutput('Stop signal sent...', 'warning');
      }
    };
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    orchestrator: Orchestrator | None = None

    async def output_callback(msg: str) -> None:
        try:
            await websocket.send_text(msg)
        except Exception:
            pass

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        settings_obj = Settings()
        orchestrator = Orchestrator(goal=goal, settings_obj=settings_obj)
        orchestrator.set_output_callback(output_callback)

        # Start the orchestrator in a background task
        orch_task = asyncio.create_task(orchestrator.run())

        # Listen for stop commands while the orchestrator runs
        async def listen_for_stop() -> None:
            try:
                while not orch_task.done():
                    try:
                        msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
                        if msg.get("action") == "stop":
                            orchestrator.request_stop()
                            await websocket.send_text("[yellow]Stop requested. Halting...[/yellow]")
                    except asyncio.TimeoutError:
                        continue
            except WebSocketDisconnect:
                pass

        listen_task = asyncio.create_task(listen_for_stop())

        done, pending = await asyncio.wait(
            {orch_task, listen_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel any pending tasks
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        if orch_task in done and orch_task.exception():
            try:
                await websocket.send_text(
                    f"[red]Orchestrator error: {orch_task.exception()}[/red]"
                )
            except WebSocketDisconnect:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        if orchestrator is not None:
            orchestrator.request_stop()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
