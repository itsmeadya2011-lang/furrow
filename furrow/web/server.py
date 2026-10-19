from __future__ import annotations

import asyncio
from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

logger = structlog.get_logger("furrow.web")

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
      ws.onerror = (err) => out.textContent += '\\nWebSocket error.\\n';
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
        model = data.get("model")

        async def send_output(message: str) -> None:
            try:
                await websocket.send_text(message)
            except Exception:
                pass

        orchestrator = Orchestrator(goal=goal, on_output=send_output)
        if model:
            orchestrator.cfg.model = model
        logger.info("furrow_run_started", goal=goal, model=model)
        await orchestrator.run()
        logger.info("furrow_run_completed", goal=goal)
    except WebSocketDisconnect:
        logger.info("furrow_websocket_disconnected")
    except Exception as e:  # noqa: BLE001
        logger.error("furrow_websocket_error", error=str(e))
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)