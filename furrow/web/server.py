from __future__ import annotations

from collections.abc import Awaitable, Callable

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

logger = structlog.get_logger(__name__)

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    max_cycles: int = 0


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <input id="model" placeholder="Model (optional)" />
    <input id="max_cycles" type="number" placeholder="Max cycles (0 = infinite)" />
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
      const msg = {goal: document.getElementById('goal').value};
      msg.model = document.getElementById('model').value;
      msg.max_cycles = parseInt(document.getElementById('max_cycles').value, 10) || 0;
      ws.send(JSON.stringify(msg));
    };
  </script>
</body>
</html>
"""
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    async def _ws_log(msg: str) -> None:
        try:
            await websocket.send_text(msg)
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001, S110
            pass

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        max_cycles = data.get("max_cycles", 0)
        if model:
            settings.model = model
        if max_cycles:
            settings.max_cycles = max_cycles
        log_callback: Callable[[str], Awaitable[None]] = _ws_log
        orchestrator = Orchestrator(goal=goal, log_callback=log_callback)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port, log_level=settings.log_level.lower())
