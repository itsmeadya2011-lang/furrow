from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from structlog import get_logger

from furrow.core.orchestrator import Orchestrator

logger = get_logger(__name__)
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
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === 'log') {
            out.textContent += data.message + '\\n';
          } else if (data.type === 'cycle') {
            out.textContent += '\\n── Cycle ' + data.cycle + ' ──\\n';
          } else {
            out.textContent += ev.data + '\\n';
          }
        } catch {
          out.textContent += ev.data + '\\n';
        }
      };
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
        logger.info("websocket_session_start", goal=goal)
        orchestrator = Orchestrator(goal=goal)

        try:
            await orchestrator.run()
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        logger.info("websocket_session_ended")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    logger.info("web_server_starting", host=host, port=port)
    uvicorn.run(app, host=host, port=port)
