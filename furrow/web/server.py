from __future__ import annotations

import asyncio
from io import StringIO
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings
from furrow.core import orchestrator as _orchestrator

app = FastAPI(title="Furrow")

__all__ = ["Settings", "StartRequest", "WebSocketConsole", "app", "run"]


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class WebSocketConsole(Console):
    """A rich ``Console`` that forwards rendered output to a WebSocket.

    Each ``print`` call is rendered to an off-screen buffer (with markup
    stripped) and the resulting plain text is forwarded to the connected
    websocket client as a text message.
    """

    def __init__(self, websocket: WebSocket, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._websocket = websocket
        self._buf = StringIO()
        self._capture = Console(
            file=self._buf,
            force_terminal=False,
            color_system=None,
            width=self.width,
        )
        self._pending: set[asyncio.Task[Any]] = set()

    def print(self, *objects: Any, **kwargs: Any) -> None:
        self._buf.seek(0)
        self._buf.truncate(0)
        self._capture.print(*objects, **kwargs)
        text = self._buf.getvalue()
        if not text:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running: cannot schedule a send, drop silently.
            return
        task = loop.create_task(self._send(text))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _send(self, text: str) -> None:
        try:
            await self._websocket.send_text(text)
        except (WebSocketDisconnect, OSError):
            # connection closed; stop broadcasting
            ...

    async def drain(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)


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
    ws_console = WebSocketConsole(websocket)
    original_console: Any = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        # Patch the orchestrator's module-level console so its console.print
        # calls are broadcast to this websocket client.
        original_console = _orchestrator.console
        _orchestrator.console = ws_console
        orchestrator = _orchestrator.Orchestrator(goal=goal)
        await orchestrator.run()
        await ws_console.drain()
        await websocket.send_json({"type": "done", "goal": goal})
    except WebSocketDisconnect:
        # Client disconnected: stop broadcasting gracefully.
        pass
    except Exception as exc:  # noqa: BLE001
        # Report unexpected errors to the client and close.
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except (WebSocketDisconnect, OSError):
            # Could not report the error to the client (likely gone); stop.
            ...
    finally:
        if original_console is not None:
            _orchestrator.console = original_console
        await ws_console.drain()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
