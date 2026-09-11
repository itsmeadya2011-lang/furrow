from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

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

        # Stream console output back to client
        original_console = orchestrator.client.__class__.__module__
        import io
        import contextlib

        buffer = io.StringIO()
        rich_console = Console(file=buffer, force_terminal=False)

        # Monkey-patch console for this session to capture output
        import furrow.core.orchestrator as orch_mod
        original_orchestrator_console = orch_mod.console
        orch_mod.console = rich_console

        async def stream_output() -> None:
            last_len = 0
            while True:
                if websocket.client_state.name == "DISCONNECTED":
                    break
                buffer.seek(last_len)
                new_text = buffer.read()
                last_len = buffer.tell()
                if new_text:
                    try:
                        await websocket.send_text(new_text)
                    except Exception:
                        break
                await asyncio.sleep(0.1)

        stream_task = asyncio.create_task(stream_output())
        try:
            await orchestrator.run()
        finally:
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
            orch_mod.console = original_orchestrator_console
            buffer.seek(0)
            remaining = buffer.read()
            if remaining:
                try:
                    await websocket.send_text(remaining)
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
