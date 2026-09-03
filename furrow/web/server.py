from __future__ import annotations

import asyncio
from typing import Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

tasks: dict[str, asyncio.Queue] = {}


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


async def _run_orchestrator(task_id: str, goal: str, queue: asyncio.Queue) -> None:
    try:
        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run()
        await queue.put("done")
    except Exception as exc:
        await queue.put(f"error: {exc}")


@app.post("/start")
async def start(req: StartRequest) -> dict:
    task_id = uuid4().hex
    queue: asyncio.Queue = asyncio.Queue()
    tasks[task_id] = queue
    asyncio.create_task(_run_orchestrator(task_id, req.goal, queue))
    return {"status": "started", "id": task_id}


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()
    queue = tasks.get(task_id)
    if queue is None:
        await websocket.close(code=1008)
        return
    await websocket.send_text("started")
    try:
        while True:
            msg = await queue.get()
            await websocket.send_text(msg)
            if msg == "done" or msg.startswith("error"):
                break
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)