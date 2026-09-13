from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

_active_orchestrators: dict[str, Orchestrator] = {}


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
  <button id="stop" style="display:none">Stop</button>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const stopBtn = document.getElementById('stop');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      stopBtn.style.display = '';
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => { out.textContent += '\\nClosed.\\n'; stopBtn.style.display = 'none'; };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
    stopBtn.onclick = async () => {
      const ws = new WebSocket('ws://' + location.host + '/ws/stop');
      ws.onopen = () => { ws.close(); };
      ws.onclose = () => { out.textContent += '\\nStopping...\\n'; };
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    orchestrator_id = f"orc_{id(websocket)}"
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)
        _active_orchestrators[orchestrator_id] = orchestrator
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    finally:
        _active_orchestrators.pop(orchestrator_id, None)


@app.websocket("/ws/stop")
async def stop_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        for orch in list(_active_orchestrators.values()):
            orch.request_stop()
        await websocket.send_json({"status": "stopping"})
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
