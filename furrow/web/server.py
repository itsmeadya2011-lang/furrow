from __future__ import annotations

import asyncio

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


class _WSOutputStream:
    """File-like object that forwards writes to a WebSocket via an asyncio Queue."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue()

    def write(self, text: str) -> int:
        if text:
            self.queue.put_nowait(text)
        return len(text)

    def flush(self) -> None:
        pass


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content='''<!DOCTYPE html>
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
</html>''')


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        ws_stream = _WSOutputStream()
        ws_console = Console(file=ws_stream, force_terminal=False, no_color=True, width=120)

        async def _drain_output() -> None:
            while True:
                text = await ws_stream.queue.get()
                await websocket.send_text(text)

        async def _run_orchestrator() -> None:
            orchestrator = Orchestrator(goal=goal, console=ws_console)
            await orchestrator.run()

        drain_task = asyncio.create_task(_drain_output())
        orch_task = asyncio.create_task(_run_orchestrator())

        try:
            await asyncio.gather(orch_task, drain_task)
        finally:
            orch_task.cancel()
            drain_task.cancel()
            for t in (orch_task, drain_task):
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        await websocket.send_text(f"\nError: {e}\n")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
