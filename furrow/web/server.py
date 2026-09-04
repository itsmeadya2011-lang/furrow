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

_active_runs: set[int] = set()


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
    body { font-family: system-ui, sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; }
    h1 { margin-bottom: 0.25em; }
    #status { color: #666; font-size: 0.9em; margin-bottom: 1em; }
    #status.running { color: #c80; }
    form { display: flex; gap: 0.5em; margin-bottom: 1em; }
    input[type=text] { flex: 1; padding: 0.5em; font-size: 1em; }
    button { padding: 0.5em 1em; font-size: 1em; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    pre { background: #1e1e1e; color: #eee; padding: 1em; border-radius: 4px;
          height: 60vh; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <div id="status">idle</div>
  <form id="form">
    <input id="goal" type="text" placeholder="Enter goal" required />
    <button id="start" type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const status = document.getElementById('status');
    const startBtn = document.getElementById('start');
    let ws = null;

    function setStatus(text, running) {
      status.textContent = text;
      status.className = running ? 'running' : '';
      startBtn.disabled = !!running;
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      setStatus('connecting...', true);
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => setStatus('running', true);
      ws.onmessage = (ev) => {
        out.textContent += ev.data + '\n';
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => setStatus('idle', false);
      ws.onerror = () => setStatus('error', false);
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
async def run_status() -> dict[str, bool]:
    return {"running": bool(_active_runs)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    run_id = id(websocket)
    _active_runs.add(run_id)
    drain_task: Optional[asyncio.Task[None]] = None
    try:
        async def drain() -> None:
            while True:
                msg = await queue.get()
                if msg is None:
                    return
                await websocket.send_text(msg)

        drain_task = asyncio.create_task(drain())

        data = await websocket.receive_json()
        req = StartRequest(**data)

        def enqueue(msg: str) -> None:
            queue.put_nowait(msg)

        client: Optional[LLMClient] = None
        if req.model:
            client_settings: Settings = default_settings.model_copy(update={"model": req.model})
            client = LLMClient(settings=client_settings)

        orchestrator = Orchestrator(goal=req.goal, client=client, output_callback=enqueue)
        await orchestrator.run()

        await websocket.send_text("[bold green]Run complete.[/bold green]")
    except WebSocketDisconnect:
        pass
    finally:
        _active_runs.discard(run_id)
        if drain_task is not None:
            await queue.put(None)
            await drain_task
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
