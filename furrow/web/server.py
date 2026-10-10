from __future__ import annotations

import asyncio
import contextlib
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


class _WebSocketStream:
    """File-like object that forwards writes to a WebSocket via a queue."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task = asyncio.ensure_future(self._drain())

    async def _drain(self) -> None:
        while True:
            text = await self._queue.get()
            try:
                await self._websocket.send_text(text)
            except WebSocketDisconnect:
                return
            self._queue.task_done()

    def write(self, text: str) -> int:
        if text:
            self._queue.put_nowait(text)
        return len(text)

    def flush(self) -> None:
        pass

    encoding = "utf-8"
    errors = "ignore"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)
        stream = _WebSocketStream(websocket)
        with contextlib.redirect_stdout(stream):
            await orchestrator.run()
        await stream._queue.join()
        await websocket.send_text('{"type": "done"}')
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
