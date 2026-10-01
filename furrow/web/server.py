from __future__ import annotations

import asyncio

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")
log = structlog.get_logger(__name__)


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class _WebSocketOutputStream:
    """A file-like object that schedules text to be sent over a WebSocket.

    Rich's ``Console`` calls ``write()`` synchronously.  Because we cannot
    ``await`` from there, we use ``asyncio.ensure_future`` to schedule the
    send on the running event loop.  The coroutine will execute at the next
    ``await`` point in the orchestrator, effectively batching output between
    async checkpoints.
    """

    def __init__(self, websocket: WebSocket):
        self._websocket = websocket

    def write(self, s: str) -> int:
        if not s or not s.strip():
            return 0
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return 0
        asyncio.ensure_future(self._send(s), loop=loop)
        return len(s)

    async def _send(self, s: str) -> None:
        try:
            await self._websocket.send_text(s)
        except WebSocketDisconnect:
            log.debug("websocket_disconnected_during_send")

    def flush(self) -> None:
        pass


class _WebSocketConsole(Console):
    """A Rich Console that streams rendered output to a WebSocket."""

    def __init__(self, websocket: WebSocket):
        stream = _WebSocketOutputStream(websocket)
        super().__init__(
            file=stream,
            width=120,
            force_terminal=False,
            no_color=True,
            highlight=False,
            soft_wrap=True,
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
        ws_console = _WebSocketConsole(websocket)
        orchestrator = Orchestrator(goal=goal, console=ws_console)
        await orchestrator.run()
        await websocket.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        await websocket.send_text(f"\nFurrow error: {e}\n")
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
