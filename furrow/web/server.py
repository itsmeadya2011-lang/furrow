from __future__ import annotations

import asyncio
import traceback

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class WebSocketFile:
    """File-like object backing a rich ``Console`` that streams output to a WebSocket."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._buffer: str = ""
        self._pending: list[asyncio.Task] = []

    def write(self, data: str) -> int:
        if not data:
            return 0
        data = data.replace("\r\n", "\n").replace("\r", "\n")
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._schedule_send(line + "\n")
        return len(data)

    def flush(self) -> None:
        if self._buffer:
            self._schedule_send(self._buffer)
            self._buffer = ""

    async def drain(self) -> None:
        self.flush()
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    def _schedule_send(self, chunk: str) -> None:
        if not chunk:
            return
        try:
            task = asyncio.create_task(self._safe_send(chunk))
        except RuntimeError:
            return
        self._pending.append(task)

    async def _safe_send(self, chunk: str) -> None:
        try:
            await self._websocket.send_text(chunk)
        except WebSocketDisconnect:
            pass


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
    ws_file = WebSocketFile(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        ws_console = Console(file=ws_file, color_system=None, width=100)
        orchestrator = Orchestrator(goal=goal, console=ws_console)
        await orchestrator.run()
        await ws_file.drain()
        await websocket.close()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        message = f"\nOrchestrator error: {exc}\n"
        trace = traceback.format_exc()
        try:
            await websocket.send_text(message + trace)
        except WebSocketDisconnect:
            pass
        try:
            await websocket.close(code=1011)
        except (WebSocketDisconnect, RuntimeError):
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
