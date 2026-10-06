from __future__ import annotations

import asyncio
import json
import structlog
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

logger = structlog.get_logger(__name__)

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


# Global orchestrator reference for WebSocket streaming
_current_orchestrator: Optional[Orchestrator] = None


class WebSocketConsole:
    """A console that writes to a WebSocket instead of stdout."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    def print(self, *args, **kwargs) -> None:
        # Convert rich markup to plain text for WebSocket
        text = " ".join(str(arg) for arg in args)
        # Strip rich markup tags (simple approach)
        import re

        text = re.sub(r"\[/?[a-zA-Z#]+\]", "", text)
        asyncio.create_task(self.websocket.send_text(text))


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 800px; margin: 2rem auto; padding: 1rem; }
        #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; height: 400px; overflow-y: auto; white-space: pre-wrap; }
        input { width: 70%; padding: 0.5rem; }
        button { padding: 0.5rem 1rem; }
    </style>
</head>
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
    let ws = null;
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      out.textContent += '\\nStarting...\\n';
      if (ws) ws.close();
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => { out.textContent += ev.data + '\\n'; out.scrollTop = out.scrollHeight; };
      ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
      ws.onopen = () => ws.send(JSON.stringify({goal}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    global _current_orchestrator
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        if model:
            from furrow.config import settings

            settings.model = model

        # Create a custom console that writes to WebSocket
        ws_console = WebSocketConsole(websocket)

        # Create orchestrator with WebSocket console
        orchestrator = Orchestrator(goal=goal)
        _current_orchestrator = orchestrator

        # Monkey-patch the console for this orchestrator run
        import furrow.core.orchestrator as orch_module

        original_console = orch_module.console
        orch_module.console = ws_console

        try:
            await orchestrator.run()
        finally:
            orch_module.console = original_console
            _current_orchestrator = None

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)