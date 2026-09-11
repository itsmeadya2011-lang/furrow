from __future__ import annotations

import asyncio
import json
from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")
log = structlog.get_logger("furrow.web")


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
      ws.onerror = (ev) => out.textContent += '\\nWebSocket error\\n';
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
        log.info("web_session_started", goal=goal)
        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run()
        log.info("web_session_completed", goal=goal)
    except WebSocketDisconnect:
        log.info("web_session_disconnected")
    except Exception as e:
        log.error("web_session_error", error=str(e))
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    log.info("web_server_starting", host=host, port=port)
    uvicorn.run(app, host=host, port=port)
