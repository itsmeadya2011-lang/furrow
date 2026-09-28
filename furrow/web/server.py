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


class QueueWriter:
    """File-like object that routes rich Console output to an asyncio Queue."""

    def __init__(self, queue: asyncio.Queue) -> None:
        self._queue = queue

    def write(self, data: str) -> int:
        if data.strip():
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.get_running_loop()
            if loop and loop.is_running():
                asyncio.create_task(self._queue.put(data))
        return len(data)

    def flush(self) -> None:
        pass


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    writer = QueueWriter(queue)
    new_console = Console(
        file=writer,
        force_terminal=False,
        no_color=True,
        width=120,
    )

    import furrow.core.orchestrator as orch_module
    original_console = orch_module.console
    orch_module.console = new_console

    orch_task: Optional[asyncio.Task] = None
    sender_task: Optional[asyncio.Task] = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)

        async def send_output() -> None:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=0.1)
                    await websocket.send_text(msg)
                except asyncio.TimeoutError:
                    if orch_task is not None and orch_task.done():
                        break
                except WebSocketDisconnect:
                    break

        orch_task = asyncio.create_task(orchestrator.run())
        sender_task = asyncio.create_task(send_output())
        await orch_task

        # Drain remaining messages after orchestrator completes
        while not queue.empty():
            msg = await queue.get()
            await websocket.send_text(msg)

        await websocket.send_text("\n[Furrow] Goal complete.\n")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"\n[error] {e}\n")
        except WebSocketDisconnect:
            pass
    finally:
        orch_module.console = original_console
        pending = [
            t for t in (orch_task, sender_task)
            if t is not None and not t.done()
        ]
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
