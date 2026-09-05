from __future__ import annotations

import asyncio
import sys
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class _WsStream:
    """File-like object that mirrors writes to stderr and an asyncio queue."""

    def __init__(self, queue: asyncio.Queue[str]) -> None:
        self._queue = queue

    def write(self, text: str) -> int:
        sys.stderr.write(text)
        if text.strip():
            self._queue.put_nowait(text)
        return len(text)

    def flush(self) -> None:
        sys.stderr.flush()


def _make_web_console(queue: asyncio.Queue[str]) -> Console:
    """Create a Console that streams output into *queue* while also echoing to stderr."""
    return Console(
        file=_WsStream(queue),
        force_terminal=False,
        no_color=True,
        highlight=False,
    )


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
      const ws = new WebSocket(location.protocol + '//' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data;
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onerror = (ev) => out.textContent += '\\n[WebSocket error]\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    ws_queue: asyncio.Queue[str] = asyncio.Queue()

    async def drain_output() -> None:
        while True:
            msg = await ws_queue.get()
            try:
                await websocket.send_text(msg)
            except WebSocketDisconnect:
                break
            except Exception:
                break
            ws_queue.task_done()

    drain_task = asyncio.create_task(drain_output())
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        web_console = _make_web_console(ws_queue)
        orchestrator = Orchestrator(goal=goal, console=web_console)

        run_task = asyncio.create_task(orchestrator.run())

        done, pending = await asyncio.wait(
            {run_task, drain_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        exc = run_task.exception() if run_task in done else None
        if exc and not isinstance(exc, WebSocketDisconnect):
            try:
                await websocket.send_text(f"\n[bold red]Error:[/bold red] {exc}\n")
            except WebSocketDisconnect:
                pass

        await websocket.close()
    except WebSocketDisconnect:
        drain_task.cancel()
        try:
            await drain_task
        except asyncio.CancelledError:
            pass
    except Exception as e:
        await websocket.send_text(f"\n[bold red]Error:[/bold red] {e}\n")
        drain_task.cancel()
        try:
            await drain_task
        except asyncio.CancelledError:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
