from __future__ import annotations

import contextlib
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


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
      ws.onerror = () => out.textContent += '\\n[error]\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Stream orchestrator progress to the connected browser in real time."""
    await websocket.accept()
    try:
        data: dict[str, Any] = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal:
            await websocket.send_text("ERROR: No goal provided.")
            await websocket.close()
            return

        async def sink(msg: str) -> None:
            """Forward orchestrator messages to the WebSocket client."""
            await websocket.send_text(msg)

        from furrow.config import settings

        orchestrator = Orchestrator(goal=goal, settings=settings, message_sink=sink)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - top-level guard for the WebSocket loop
        with contextlib.suppress(Exception):
            await websocket.send_text(f"ERROR: {exc}")
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
