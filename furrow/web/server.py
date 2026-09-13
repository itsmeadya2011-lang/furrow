from __future__ import annotations

import asyncio
import structlog
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

logger = structlog.get_logger()
app = FastAPI(title="Furrow")

_active_tasks: dict[str, asyncio.Task] = {}


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
    task_id = id(asyncio.current_task())
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        logger.info("websocket_start", goal=goal[:100], task_id=task_id)
        orchestrator = Orchestrator(goal=goal)
        asyncio_task = asyncio.create_task(orchestrator.run())
        _active_tasks[task_id] = asyncio_task
        try:
            await asyncio_task
        except asyncio.CancelledError:
            logger.info("websocket_cancelled", task_id=task_id)
            raise
        finally:
            _active_tasks.pop(task_id, None)
    except WebSocketDisconnect:
        logger.info("websocket_disconnect", task_id=task_id)
    except Exception as e:
        logger.error("websocket_error", task_id=task_id, error=str(e))


@app.on_event("shutdown")
async def shutdown() -> None:
    logger.info("shutdown_start", active_tasks=len(_active_tasks))
    for task_id, task in list(_active_tasks.items()):
        task.cancel()
    if _active_tasks:
        await asyncio.gather(*_active_tasks.values(), return_exceptions=True)
    logger.info("shutdown_complete")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    logger.info("server_start", host=host, port=port)
    uvicorn.run(app, host=host, port=port)
