from __future__ import annotations

import contextlib
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator
from furrow.logging import configure_logging, get_logger

configure_logging()
log = get_logger("web")

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
      out.textContent = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => {
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
      ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        const line = `[${data.event || 'info'}] ${data.data ? JSON.stringify(data.data) : ''}`;
        out.textContent += line + '\n';
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\nClosed.\n';
      ws.onerror = () => out.textContent += '\nError connecting.\n';
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    log.info("websocket_connected")
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        async def on_event(event: str, event_data: Any) -> None:
            try:
                await websocket.send_json({"event": event, "data": event_data})
            except Exception:
                log.warning("send_failed", event_name=event)

        orchestrator = Orchestrator(goal=goal, on_event=on_event)
        await orchestrator.run()
        await websocket.send_json({"event": "finished", "data": {"cycles": orchestrator.cycles}})
    except WebSocketDisconnect:
        log.info("websocket_disconnected")
    except Exception as e:
        log.error("websocket_error", error=str(e))
        with contextlib.suppress(Exception):
            await websocket.send_json({"event": "error", "data": str(e)})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
