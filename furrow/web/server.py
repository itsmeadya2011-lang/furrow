from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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
    orchestrator: Orchestrator | None = None
    orch_task: asyncio.Task | None = None
    hb_task: asyncio.Task | None = None

    async def heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(2.0)
                if orchestrator is None:
                    continue
                await websocket.send_json(
                    {"type": "event", "event": "heartbeat", "cycles": orchestrator.cycles}
                )
        except (asyncio.CancelledError, WebSocketDisconnect):
            return

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)
        hb_task = asyncio.create_task(heartbeat())
        try:
            orch_task = asyncio.create_task(orchestrator.run())
            await orch_task
            await websocket.send_json(
                {"type": "event", "event": "done", "cycles": orchestrator.cycles}
            )
        except Exception as exc:
            await websocket.send_json(
                {"type": "event", "event": "error", "message": str(exc)}
            )
        finally:
            if hb_task and not hb_task.done():
                hb_task.cancel()
    except WebSocketDisconnect:
        if orch_task and not orch_task.done():
            orch_task.cancel()
        if hb_task and not hb_task.done():
            hb_task.cancel()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
