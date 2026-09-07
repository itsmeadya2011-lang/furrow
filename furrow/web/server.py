from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, settings as default_settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")

active_orchestrator: Optional[Orchestrator] = None
stop_event = asyncio.Event()


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None
    max_cycles: Optional[int] = None
    no_test: bool = False


class StopRequest(BaseModel):
    pass


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
    h1 { color: #2e7d32; }
    #form { margin: 1rem 0; }
    #goal { width: 70%; padding: 0.5rem; }
    button { padding: 0.5rem 1rem; cursor: pointer; }
    #stop { background: #c62828; color: white; border: none; margin-left: 0.5rem; }
    #status { margin: 0.5rem 0; font-weight: bold; }
    #out { background: #f5f5f5; padding: 1rem; border-radius: 4px; white-space: pre-wrap; max-height: 70vh; overflow-y: auto; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <div id="status">Status: idle</div>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
    <button type="button" id="stop" disabled>Stop</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const status = document.getElementById('status');
    const stopBtn = document.getElementById('stop');
    let ws = null;

    function setStatus(s) { status.textContent = 'Status: ' + s; }

    stopBtn.onclick = async () => {
      if (!ws) return;
      setStatus('stopping...');
      await fetch('/stop', {method: 'POST'});
    };

    form.onsubmit = async (e) => {
      e.preventDefault();
      if (ws) { ws.close(); }
      out.textContent = '';
      setStatus('starting...');
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(proto + '//' + location.host + '/ws');
      ws.onopen = () => {
        setStatus('running');
        stopBtn.disabled = false;
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
      ws.onmessage = (ev) => { out.textContent += ev.data + '\\n'; out.scrollTop = out.scrollHeight; };
      ws.onclose = () => { setStatus('idle'); stopBtn.disabled = true; out.textContent += '\\nClosed.\\n'; };
      ws.onerror = () => { setStatus('error'); stopBtn.disabled = true; };
    };
  </script>
</body>
</html>
""")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "running": active_orchestrator is not None}


@app.post("/stop")
async def stop_orchestrator() -> dict:
    global active_orchestrator
    if active_orchestrator is None:
        return {"status": "no_running_orchestrator"}
    stop_event.set()
    return {"status": "stop_signal_sent"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    global active_orchestrator
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        modified_settings = Settings(
            model=data.get("model", default_settings.model),
            max_cycles=int(data.get("max_cycles", default_settings.max_cycles)),
        )
        client = LLMClient(settings=modified_settings)

        def on_output(message: str) -> None:
            asyncio.create_task(websocket.send_text(message))

        orchestrator = Orchestrator(
            goal=goal,
            client=client,
            max_cycles=modified_settings.max_cycles,
            no_test=modified_settings.no_test,
            on_output=on_output,
            stop_event=stop_event,
        )
        active_orchestrator = orchestrator
        stop_event.clear()
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    finally:
        active_orchestrator = None
        stop_event.clear()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
