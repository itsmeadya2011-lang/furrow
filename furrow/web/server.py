from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import LogEvent, Orchestrator, OrchestratorStop
from furrow.logging import configure_logging

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""<!DOCTYPE html>
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
        const data = JSON.parse(ev.data);
        out.textContent += '[' + data.level + '] ' + data.message + '\\n';
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>"""
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    goal = ""
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
    except WebSocketDisconnect:
        return

    async def stream(event: LogEvent) -> None:
        await websocket.send_json(
            {"level": event.level, "message": event.message}
        )

    orchestrator = Orchestrator(goal=goal, listeners=[stream])
    try:
        await orchestrator.run()
    except OrchestratorStop:
        pass
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001 - intentional top-level error reporter for the client
        # Top-level safety net: report *any* orchestration error to the client.
        try:
            await websocket.send_json({"level": "error", "message": str(e)})
        except WebSocketDisconnect:
            pass  # client already gone; nothing more to do


def run(host: str = "0.0.0.0", port: int = 8000, log_level: str = "INFO") -> None:
    configure_logging(log_level)
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
