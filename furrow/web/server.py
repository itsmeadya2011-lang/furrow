from __future__ import annotations

import json

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/health")
async def health() -> dict:
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
    except WebSocketDisconnect:
        return

    goal = data.get("goal", "")
    model = data.get("model")
    ws_settings = settings
    if model:
        ws_settings = settings.model_copy(update={"model": model})

    orch = Orchestrator(goal=goal, settings=ws_settings)

    async def emit(event: str, payload: dict) -> None:
        await websocket.send_text(json.dumps({"event": event, "data": payload}))

    orch.on_event = emit
    await websocket.send_text(json.dumps({"event": "start", "data": {"goal": goal}}))

    try:
        await orch.run()
    except WebSocketDisconnect:
        # Client disconnected mid-run; stop the orchestrator.
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await websocket.send_text(json.dumps({"event": "error", "data": {"error": str(e)}}))
        except Exception:  # noqa: BLE001, S110
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)