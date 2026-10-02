from __future__ import annotations

import asyncio
import json

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

logger = structlog.get_logger()
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
    await websocket.accept()
    logger.info("websocket_connected")

    task: asyncio.Task | None = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        if not goal:
            await websocket.send_text(json.dumps({"error": "No goal provided"}))
            await websocket.close()
            return

        orchestrator = Orchestrator(goal=goal)

        async def ws_handler(line: str) -> None:
            """Forward orchestrator output to the WebSocket client."""
            try:
                await websocket.send_text(line)
            except Exception as e:  # noqa: BLE001
                logger.warning("websocket_send_failed", error=str(e))

        orchestrator.set_output_handler(ws_handler)

        # Run the orchestrator; cancel cleanly if the client disconnects.
        task = asyncio.create_task(orchestrator.run())
        await task

        await websocket.send_text("[DONE]")
        logger.info("orchestrator_finished")
    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    except Exception as e:  # noqa: BLE001
        logger.error("websocket_error", error=str(e))
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:  # noqa: S110, BLE001
            pass
    finally:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await websocket.close()
        logger.info("websocket_closed")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
