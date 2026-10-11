from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class WebSocketConsole(Console):
    """A rich Console that streams everything printed to it to a WebSocket client."""

    def __init__(self, websocket: WebSocket, **kwargs: Any) -> None:
        kwargs.setdefault("no_color", True)
        super().__init__(**kwargs)
        self._websocket = websocket
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._pending: set[concurrent.futures.Future] = set()

    def print(self, *args: Any, **kwargs: Any) -> None:
        with self.capture() as capture:
            super().print(*args, **kwargs)
        message = capture.get()
        if message.strip():
            self._schedule_send(message)

    def _schedule_send(self, message: str) -> None:
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.get_event_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._websocket.send_text(message), self._loop
        )
        self._pending.add(future)
        future.add_done_callback(self._pending.discard)

    async def flush(self) -> None:
        while self._pending:
            future = self._pending.pop()
            try:
                await asyncio.wrap_future(future)
            except (RuntimeError, OSError):
                pass


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


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
    ws_console = WebSocketConsole(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal, console=ws_console)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    finally:
        await ws_console.flush()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
