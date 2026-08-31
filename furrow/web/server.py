from __future__ import annotations

import asyncio
import structlog
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

logger = structlog.get_logger("furrow.web.server")

app = FastAPI(title="Furrow")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connections: list[WebSocket] = []


class StartRequest(BaseModel):
    goal: str
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
    active_connections.append(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run()
    except WebSocketDisconnect:
        logger.info("websocket_disconnected", message="Client disconnected")
    except Exception as exc:
        logger.error("orchestrator_error", error=str(exc), exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("shutdown_started", message="Closing active WebSocket connections")
    close_tasks = [connection.close() for connection in active_connections]
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)
    active_connections.clear()
    logger.info("shutdown_complete", message="All connections closed")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
