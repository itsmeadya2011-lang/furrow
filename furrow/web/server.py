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


class ProgressBus:
    def __init__(self) -> None:
        self._subscribers: list[WebSocket] = []

    def subscribe(self, ws: WebSocket) -> None:
        self._subscribers.append(ws)

    def unsubscribe(self, ws: WebSocket) -> None:
        if ws in self._subscribers:
            self._subscribers.remove(ws)

    async def publish(self, message: str) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._subscribers):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unsubscribe(ws)


bus = ProgressBus()


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
  body { font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
  #log { background: #111; color: #eee; padding: 1rem; border-radius: 6px; height: 50vh; overflow-y: auto; font-family: monospace; white-space: pre-wrap; }
  input { width: 70%; padding: 0.5rem; }
  button { padding: 0.5rem 1rem; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <div id="log"></div>
  <script>
    const form = document.getElementById('form');
    const log = document.getElementById('log');
    form.onsubmit = async (e) => {
      e.preventDefault();
      log.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => { log.textContent += ev.data + '\\n'; log.scrollTop = log.scrollHeight; };
      ws.onclose = () => { log.textContent += '\\nClosed.\\n'; };
      ws.onerror = () => { log.textContent += '\\nConnection error.\\n'; };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    bus.subscribe(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        await bus.publish(f"Goal received: {goal}\n")
        await bus.publish("Starting orchestrator...\n")
        orchestrator = Orchestrator(goal=goal)
        task = asyncio.create_task(orchestrator.run())
        done, pending = await asyncio.wait(
            {task, asyncio.create_task(websocket.receive_text())},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            await bus.publish("Orchestrator finished.\n")
            for p in pending:
                p.cancel()
        else:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
