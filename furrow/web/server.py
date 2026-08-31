from __future__ import annotations

import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError

from furrow.config import Settings, get_settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)
app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=1000)
    model: Optional[str] = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
        try:
            request = StartRequest(**data)
        except ValidationError as e:
            await websocket.send_json({"error": f"Invalid request: {e}"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        if not request.goal.strip():
            await websocket.send_json({"error": "Goal cannot be empty"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        settings = get_settings()
        if request.model:
            settings.model = request.model
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal=request.goal, client=client, max_cycles=settings.max_cycles)
        try:
            await orchestrator.run()
            await websocket.send_json({"status": "complete"})
        except Exception as e:
            logger.exception("Orchestrator failed")
            await websocket.send_json({"error": f"Orchestration failed: {str(e)}"})
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
