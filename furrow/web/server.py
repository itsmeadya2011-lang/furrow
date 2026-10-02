from __future__ import annotations

import asyncio

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow import configure_logging
from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

logger = structlog.get_logger(__name__)

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


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
      ws.onerror = (ev) => { out.textContent += '\\nError: ' + ev.message + '\\n'; };
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
        if not goal:
            await websocket.send_json({"type": "error", "message": "No goal provided"})
            return
        settings = Settings()
        if data.get("model"):
            settings.model = data["model"]
        orchestrator = Orchestrator(goal=goal, settings=settings)
        await orchestrator.run()
        await websocket.send_json({"type": "complete", "message": "Furrow run finished"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except asyncio.CancelledError:
        logger.info("WebSocket task cancelled")
        raise
    except Exception as e:  # noqa: BLE001 - top-level websocket handler reports errors to client
        logger.error("WebSocket error", error=str(e))
        await websocket.send_json({"type": "error", "message": str(e)})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    configure_logging()
    logger.info("Starting Furrow web server", host=host, port=port)
    uvicorn.run(app, host=host, port=port)
