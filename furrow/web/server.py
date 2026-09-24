from __future__ import annotations

import asyncio
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, settings, configure_logging
from furrow.core.orchestrator import Orchestrator

configure_logging()

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
        
        # Send updates back through websocket
        original_print = print
        def ws_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            asyncio.create_task(websocket.send_text(msg))
            original_print(*args, **kwargs)
        
        # Monkey patch console print for this session
        import furrow.core.orchestrator as orchestrator_module
        original_console = orchestrator_module.console
        
        class WSConsole:
            def print(self, *args, **kwargs):
                msg = " ".join(str(a) for a in args)
                asyncio.create_task(websocket.send_text(msg))
                original_console.print(*args, **kwargs)
        
        orchestrator_module.console = WSConsole()
        
        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"Error: {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
