from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class WebSocketConsole:
    """A file-like object whose write() queues text to be sent over a WebSocket."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._drain_task: asyncio.Task | None = None

    def start(self) -> None:
        self._drain_task = asyncio.get_running_loop().create_task(self._drain())

    def write(self, text: str) -> int:
        if text:
            self._queue.put_nowait(text)
        return len(text)

    def flush(self) -> None:
        return None

    async def _drain(self) -> None:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            await self.websocket.send_text(chunk)

    async def close(self) -> None:
        await self._queue.put(None)
        if self._drain_task is not None:
            await self._drain_task


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
        ws_console = WebSocketConsole(websocket)
        ws_console.start()
        try:
            orchestrator = Orchestrator(goal=goal, console=Console(file=ws_console))
            await orchestrator.run()
        finally:
            await ws_console.close()
        await websocket.send_text('{"type": "done"}')
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
