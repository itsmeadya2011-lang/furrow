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


class WebSocketOutput:
    def __init__(self, queue: asyncio.Queue) -> None:
        self.queue = queue

    def write(self, text: str) -> None:
        if text.strip():
            self.queue.put_nowait(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


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
    sender_task: asyncio.Task | None = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        queue: asyncio.Queue = asyncio.Queue()
        ws_console = Console(file=WebSocketOutput(queue), force_terminal=False)
        orchestrator = Orchestrator(goal=goal, console=ws_console)

        async def forward_to_websocket() -> None:
            while True:
                msg = await queue.get()
                await websocket.send_text(msg)

        sender_task = asyncio.create_task(forward_to_websocket())
        await orchestrator.run()
        await websocket.send_text("Session complete.")
    except WebSocketDisconnect:
        pass
    finally:
        if sender_task is not None:
            sender_task.cancel()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
