from __future__ import annotations

import asyncio
from io import StringIO
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

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


class _StreamConsole:
    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._buffer = StringIO()
        self._console = Console(file=self._buffer, force_terminal=False)

    def print(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self._console.print(*args, **kwargs)
        self._flush()

    def _flush(self) -> None:
        text = self._buffer.getvalue()
        if text:
            try:
                asyncio.get_running_loop().create_task(self._ws.send_text(text))
            except RuntimeError:
                pass
            self._buffer.seek(0)
            self._buffer.truncate(0)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        settings = Settings()
        if model:
            settings.model = model

        stream_console = _StreamConsole(websocket)

        # Patch console output for streaming
        import furrow.core.orchestrator as orchestrator_module
        original_console = orchestrator_module.console
        orchestrator_module.console = stream_console

        try:
            orchestrator = Orchestrator(goal=goal)
            await orchestrator.run()
        finally:
            orchestrator_module.console = original_console
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
