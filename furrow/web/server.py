from __future__ import annotations

import asyncio

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")
logger = structlog.get_logger("furrow.web")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


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
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    client = websocket.client
    logger.info(
        "websocket_connect",
        host=getattr(client, "host", None),
        port=getattr(client, "port", None),
    )
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        logger.info("websocket_goal_received", goal=goal)

        async def send_progress(msg: str) -> None:
            try:
                await websocket.send_text(msg)
            except WebSocketDisconnect:
                logger.info("websocket_disconnect_during_run")
            except Exception as exc:  # noqa: BLE001
                logger.error("websocket_send_failed", error=str(exc))

        pending: set[asyncio.Task[None]] = set()

        def on_progress(msg: str) -> None:
            task = asyncio.create_task(send_progress(msg))
            pending.add(task)
            task.add_done_callback(pending.discard)

        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run(on_progress=on_progress)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        logger.info("orchestrator_run_complete")
    except WebSocketDisconnect:
        logger.info("websocket_disconnect")
    except Exception as exc:  # noqa: BLE001
        logger.error("websocket_error", error=str(exc))
        try:
            await websocket.send_text(f"Error: {exc}")
        except WebSocketDisconnect:
            logger.info("websocket_disconnect_during_error")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
