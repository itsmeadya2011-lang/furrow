from __future__ import annotations

import asyncio
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


INDEX_HTML = """<!DOCTYPE html>
<html>
<head><title>Furrow</title><meta charset="utf-8">
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; }
  #log { white-space: pre-wrap; font-family: monospace; background: #f6f8fa; padding: 1rem; border-radius: 6px; height: 70vh; overflow: auto; }
  button { padding: .4rem .8rem; }
  input#goal { width: 60ch; padding: .3rem; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <pre id="log"></pre>
  <script>
    const form = document.getElementById('form');
    const log = document.getElementById('log');
    form.onsubmit = async (e) => {
      e.preventDefault();
      log.textContent += '\nStarting...\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        let msg;
        try { msg = JSON.parse(ev.data); } catch {
          log.textContent += ev.data + '\n';
          return;
        }
        const ts = new Date().toISOString().slice(11, 19);
        const extra = [];
        if (msg.description) extra.push(msg.description);
        if (msg.summary) extra.push(msg.summary);
        if (msg.result) extra.push(String(msg.result).slice(0, 120));
        if (msg.error) extra.push('error: ' + msg.error);
        if (msg.tasks) log.textContent += '  tasks: ' + JSON.stringify(msg.tasks) + '\n';
        log.textContent += '[' + ts + '] ' + msg.event + (extra.length ? ': ' + extra.join(' | ') : '') + '\n';
        log.scrollTop = log.scrollHeight;
      };
      ws.onclose = () => log.textContent += '\nClosed.\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        settings = Settings()
        if data.get("model"):
            settings.model = data["model"]

        events: "asyncio.Queue[Optional[dict[str, Any]]]" = asyncio.Queue()

        async def drain() -> None:
            while True:
                msg = await events.get()
                if msg is None:
                    events.task_done()
                    break
                try:
                    await websocket.send_json(msg)
                except WebSocketDisconnect:
                    break
                events.task_done()

        drain_task = asyncio.create_task(drain())

        def emit(name: str, payload: Optional[dict] = None) -> None:
            asyncio.create_task(events.put({"event": name, **(payload or {})}))

        orchestrator = Orchestrator(
            goal=goal,
            client=LLMClient(settings=settings),
            on_event=emit,
        )
        await orchestrator.run()
        await events.put(None)
        await asyncio.wait_for(drain_task, timeout=5)
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
