from __future__ import annotations

import asyncio
from typing import Callable, Optional

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
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Furrow</title>
  <style>
    :root { color-scheme: light dark; }
    body {
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      max-width: 820px; margin: 0 auto; padding: 24px;
      background: #0f1117; color: #e6e6e6;
    }
    h1 { margin: 0 0 16px; }
    form { display: flex; gap: 8px; margin-bottom: 12px; }
    input[type="text"] {
      flex: 1; padding: 10px 12px; border-radius: 8px;
      border: 1px solid #2a2f3a; background: #161a22; color: inherit;
    }
    button {
      padding: 10px 16px; border-radius: 8px; border: 0; cursor: pointer;
      background: #3b82f6; color: #fff; font-weight: 600;
    }
    button:disabled { opacity: 0.5; cursor: default; }
    button.stop { background: #ef4444; }
    #status { font-size: 0.85rem; color: #9aa4b2; margin-bottom: 8px; min-height: 1.2em; }
    #out {
      height: 60vh; overflow-y: auto; padding: 12px;
      border-radius: 8px; background: #161a22; border: 1px solid #2a2f3a;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85rem;
      white-space: pre-wrap; word-break: break-word;
    }
    .msg { padding: 2px 0; border-bottom: 1px solid #1c2230; }
    .ts { color: #6b7280; margin-right: 8px; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" type="text" placeholder="Enter goal" required />
    <button type="submit" id="start">Start</button>
    <button type="button" id="stop" class="stop" disabled>Stop</button>
  </form>
  <div id="status">Idle. Enter a goal and press Start.</div>
  <div id="out"></div>
  <script>
    const form = document.getElementById('form');
    const goalInput = document.getElementById('goal');
    const startBtn = document.getElementById('start');
    const stopBtn = document.getElementById('stop');
    const statusEl = document.getElementById('status');
    const out = document.getElementById('out');
    let ws = null;

    function addMessage(text) {
      const div = document.createElement('div');
      div.className = 'msg';
      const ts = document.createElement('span');
      ts.className = 'ts';
      ts.textContent = new Date().toLocaleTimeString();
      div.appendChild(ts);
      div.appendChild(document.createTextNode(text));
      out.appendChild(div);
      out.scrollTop = out.scrollHeight;
    }

    function setRunning(running) {
      startBtn.disabled = running;
      stopBtn.disabled = !running;
      goalInput.disabled = running;
      statusEl.textContent = running ? 'Connecting / running…' : 'Idle.';
    }

    form.addEventListener('submit', (e) => {
      e.preventDefault();
      out.innerHTML = '';
      setRunning(true);
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(proto + '://' + location.host + '/ws');
      ws.onopen = () => {
        statusEl.textContent = 'Connected.';
        ws.send(JSON.stringify({goal: goalInput.value}));
      };
      ws.onmessage = (ev) => addMessage(ev.data);
      ws.onclose = () => {
        addMessage('— connection closed —');
        setRunning(false);
        ws = null;
      };
      ws.onerror = () => { statusEl.textContent = 'Connection error.'; };
    });

    stopBtn.addEventListener('click', () => {
      if (ws) { ws.close(); }
      setRunning(false);
    });
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[str] = asyncio.Queue()
    run_task: asyncio.Task | None = None

    async def sender() -> None:
        while True:
            message = await queue.get()
            await websocket.send_text(message)

    def on_update(message: str) -> None:
        queue.put_nowait(message)

    sender_task = asyncio.create_task(sender())

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal, on_update=on_update)
        run_task = asyncio.create_task(orchestrator.run())
        await run_task
        await queue.put("Done.")
    except WebSocketDisconnect:
        if run_task is not None:
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
    finally:
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
