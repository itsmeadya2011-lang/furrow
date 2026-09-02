from __future__ import annotations

import asyncio
import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator

log = logging.getLogger(__name__)

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
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        run_settings = Settings(**settings.model_dump())
        for key in ("provider", "model", "max_parallel_tasks", "max_cycles", "workspace"):
            value = data.get(key)
            if value is not None:
                setattr(run_settings, key, value)

        async def on_event(event: str, payload: dict) -> None:
            try:
                await websocket.send_json({"event": event, "data": payload})
                await websocket.send_text(f"[{event}] {payload}")
            except Exception:
                pass

        orchestrator = Orchestrator(goal=goal, on_event=on_event)
        try:
            await orchestrator.run()
        except Exception as exc:
            log.exception("orchestrator run failed")
            try:
                await websocket.send_json({"event": "error", "data": {"message": str(exc)}})
            except Exception:
                pass
        finally:
            try:
                await websocket.send_json({"event": "complete"})
            except Exception:
                pass
            await websocket.close()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)