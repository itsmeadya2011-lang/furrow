from __future__ import annotations

import asyncio
import re
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class QueueWriter:
    """File-like object that routes rich console output into an asyncio.Queue."""

    _ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

    def __init__(self, queue: asyncio.Queue) -> None:
        self._queue = queue
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        text = self._ANSI_RE.sub("", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._queue.put_nowait(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._queue.put_nowait(self._buffer)
            self._buffer = ""

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return "utf-8"


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
    except WebSocketDisconnect:
        return

    goal = data.get("goal", "")
    queue: asyncio.Queue = asyncio.Queue()
    writer = QueueWriter(queue)
    ws_console = Console(
        file=writer,
        force_terminal=False,
        color_system=None,
        width=100,
    )
    orchestrator = Orchestrator(goal=goal, console=ws_console)

    async def drain() -> None:
        while True:
            line = await queue.get()
            if line is None:
                break
            await websocket.send_text(line)

    async def run_and_signal() -> None:
        try:
            await orchestrator.run()
        finally:
            await queue.put(None)

    run_task = asyncio.create_task(run_and_signal())
    drain_task = asyncio.create_task(drain())
    try:
        await asyncio.gather(run_task, drain_task)
    except WebSocketDisconnect:
        pass
    finally:
        run_task.cancel()
        drain_task.cancel()
        await asyncio.gather(run_task, drain_task, return_exceptions=True)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
