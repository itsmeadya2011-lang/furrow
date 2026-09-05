from __future__ import annotations

import asyncio
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


class WebSocketWriter:
    """Sync file-like object that queues writes for async WebSocket sending."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    def write(self, text: str) -> int:
        self._queue.put_nowait(text)
        return len(text)

    def flush(self) -> None:
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
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        writer = WebSocketWriter(websocket)
        console = Console(file=writer, force_terminal=False)

        orchestrator = Orchestrator(goal=goal)
        # Patch the orchestrator's console to our websocket-aware one
        import furrow.core.orchestrator as orch_mod
        orch_mod.console = console

        # Drain queued output while orchestrator runs
        send_task = asyncio.create_task(_drain_queue(writer, websocket))
        try:
            await orchestrator.run()
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
    except WebSocketDisconnect:
        pass


async def _drain_queue(writer: WebSocketWriter, websocket: WebSocket) -> None:
    while True:
        text = await writer._queue.get()
        try:
            await websocket.send_text(text)
        except Exception:
            break


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
