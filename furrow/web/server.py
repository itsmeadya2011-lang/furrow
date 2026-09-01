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
    max_cycles: Optional[int] = None


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
    orchestrator: Optional[Orchestrator] = None
    try:
        raw = await websocket.receive_text()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"goal": raw}
        goal = (data.get("goal") or "").strip()
        if not goal:
            await websocket.send_text("error: empty goal")
            await websocket.close()
            return
        settings = Settings()
        if data.get("max_cycles") is not None:
            try:
                settings.max_cycles = int(data["max_cycles"])
            except (TypeError, ValueError):
                pass
        orchestrator = Orchestrator(goal=goal, settings=settings)
        try:
            await orchestrator.run()
        except WebSocketDisconnect:
            orchestrator.stop()
            raise
    except WebSocketDisconnect:
        if orchestrator is not None:
            orchestrator.stop()


@app.post("/start")
async def start(req: StartRequest) -> dict[str, str]:
    """Fire-and-forget endpoint that kicks off a run asynchronously."""
    settings = Settings()
    if req.max_cycles is not None:
        settings.max_cycles = req.max_cycles
    orchestrator = Orchestrator(goal=req.goal, settings=settings)
    asyncio.create_task(orchestrator.run())
    return {"status": "started", "goal": req.goal}


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
