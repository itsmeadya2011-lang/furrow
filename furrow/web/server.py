from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


INDEX_HTML = """
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
      out.textContent = '';
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onerror = (ev) => out.textContent += '\\nError: ' + ev.message + '\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.post("/start")
async def start(request: StartRequest) -> dict:
    """Start a run synchronously over HTTP (best-effort, limited)."""
    if request.model:
        from furrow.config import settings

        settings.model = request.model
    orchestrator = Orchestrator(goal=request.goal)
    asyncio.create_task(orchestrator.run())
    return {"status": "started", "goal": request.goal}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        if model:
            from furrow.config import settings

            settings.model = model

        async def send(message: str) -> None:
            await websocket.send_text(message)

        orchestrator = Orchestrator(goal=goal, on_message=send)
        await orchestrator.run()
        await websocket.send_text(
            f"\nFurrow finished. Cycles: {orchestrator.cycles}, Completed: {orchestrator.completed}"
        )
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - report any run failure to the client
        try:
            await websocket.send_text(f"\nError: {exc}")
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
