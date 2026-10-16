from __future__ import annotations

import asyncio
import io
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketConsole(Console):
    """A Rich Console that forwards rendered output to an asyncio.Queue."""

    def __init__(self, queue: asyncio.Queue[str]) -> None:
        super().__init__(file=io.StringIO(), force_terminal=False)
        self.queue = queue
        self._buffer = ""

    def _write(self, text: str) -> None:
        self._buffer += text
        # Process complete lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                try:
                    self.queue.put_nowait(line)
                except asyncio.QueueFull:
                    pass

    def print(self, *objects, sep: str = " ", end: str = "\n", style=None, **kwargs) -> None:
        # Render to string, then forward each line to the queue
        with self.capture() as capture:
            super().print(*objects, sep=sep, end=end, style=style, **kwargs)
        text = capture.get()
        for line in text.splitlines():
            if line.strip():
                try:
                    self.queue.put_nowait(line)
                except asyncio.QueueFull:
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
    let ws = null;
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      const goal = document.getElementById('goal').value;
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(protocol + '//' + location.host + '/ws');
      ws.onmessage = (ev) => { out.textContent += ev.data + '\\n'; out.scrollTop = out.scrollHeight; };
      ws.onclose = () => { out.textContent += '\\nClosed.\\n'; ws = null; };
      ws.onerror = () => { out.textContent += '\\nError.\\n'; };
      ws.send(JSON.stringify({goal: goal}));
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

        queue: asyncio.Queue[str] = asyncio.Queue()
        console = WebSocketConsole(queue)

        if model:
            from furrow.config import settings
            settings.model = model

        orchestrator = Orchestrator(goal=goal, console=console)

        # Drain queue and send to websocket while orchestrator runs
        async def drain_queue() -> None:
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                try:
                    await websocket.send_text(msg)
                except Exception:
                    break

        drain_task = asyncio.create_task(drain_queue())
        try:
            await orchestrator.run()
        finally:
            await queue.put(None)
            await drain_task
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
