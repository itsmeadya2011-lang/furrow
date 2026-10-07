from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core import orchestrator as orch_module
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str


class WebSocketConsoleTransport:
    """File-like object for rich.Console that queues output for async WebSocket delivery."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()

    def write(self, s: str) -> None:
        if s:
            self.queue.put_nowait(s)

    def flush(self) -> None:
        pass

    async def drain(self) -> None:
        while True:
            data = await self.queue.get()
            if data is None:
                break
            await self.websocket.send_text(data)


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

        transport = WebSocketConsoleTransport(websocket)
        console = Console(file=transport, force_terminal=False)

        original_console = orch_module.console
        orch_module.console = console
        try:
            orchestrator = Orchestrator(goal=goal)

            async def run_with_drain() -> None:
                drain_task = asyncio.create_task(transport.drain())
                try:
                    await orchestrator.run()
                finally:
                    await transport.queue.put(None)
                    await drain_task

            await run_with_drain()
        finally:
            orch_module.console = original_console
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)