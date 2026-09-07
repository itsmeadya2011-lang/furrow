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

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


@app.get("/")
async def index() -> HTMLResponse:
    logger.info("web_index_served")
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  h1 { color: #333; }
  #form { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
  #goal { flex: 1; padding: 0.5rem; font-size: 1rem; }
  button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
  #out { background: #f4f4f4; padding: 1rem; border-radius: 4px; white-space: pre-wrap; font-family: monospace; }
</style>
</head>
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
    logger.info("web_websocket_connected")
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        orchestrator = Orchestrator(goal=goal)
        if model:
            from furrow.config import settings
            settings.model = model
        await orchestrator.run()
    except WebSocketDisconnect:
        logger.info("web_websocket_disconnected")
    except Exception as e:
        logger.error("web_websocket_error", error=str(e))


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
