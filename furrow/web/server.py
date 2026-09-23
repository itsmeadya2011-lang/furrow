from __future__ import annotations

import asyncio
import json
from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, configure_logging
from furrow.core.orchestrator import Orchestrator

logger = structlog.get_logger()

app = FastAPI(title="Furrow")


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
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        if model:
            from furrow.config import settings
            settings.model = model
        
        orchestrator = Orchestrator(goal=goal)
        
        # Override console output to send via websocket
        original_print = print
        def ws_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            asyncio.create_task(websocket.send_text(msg))
            original_print(*args, **kwargs)
        
        # Monkey patch for this session
        import builtins
        builtins.print = ws_print
        
        try:
            await orchestrator.run()
        finally:
            builtins.print = original_print
            
    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
        pass


def run(host: str = "0.0.0.0", port: int = 8000, log_level: str = "INFO") -> None:
    configure_logging(log_level)
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
