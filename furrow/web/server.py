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

_STOP_SENTINEL = "__FURROW_STOP__"


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[str] = asyncio.Queue()
    drain_task: Optional[asyncio.Task] = None

    async def _drain() -> None:
        while True:
            msg = await queue.get()
            if msg is _STOP_SENTINEL:
                break
            try:
                await websocket.send_text(msg)
            except WebSocketDisconnect:
                break

    drain_task = asyncio.create_task(_drain())

    def on_output(msg: str) -> None:
        queue.put_nowait(msg)

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        orch_settings = Settings()
        if model:
            orch_settings.model = model
            orch_settings.planner_model = model
            orch_settings.worker_model = model
            orch_settings.tester_model = model

        orchestrator = Orchestrator(
            goal=goal, settings=orch_settings, output_callback=on_output
        )
        await orchestrator.run()
    except WebSocketDisconnect:
        if drain_task is not None:
            drain_task.cancel()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass
        return

    queue.put_nowait(_STOP_SENTINEL)
    await drain_task
    try:
        await websocket.close()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
