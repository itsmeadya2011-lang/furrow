from __future__ import annotations

import asyncio
from io import StringIO
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


class WebSocketConsole:
    """File-like object that captures console output and streams it over a WebSocket.

    Rich's Console writes through a file-like object's ``write``/``flush`` methods.
    This implementation buffers partial lines and sends complete lines to the
    client, which keeps the wire protocol clean and avoids half-line messages.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self._buffer = StringIO()
        self._closed = False

    async def _send_line(self, line: str) -> None:
        if self._closed:
            return
        try:
            await self.websocket.send_text(line)
        except Exception:
            self._closed = True

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer.write(text)
        # Send any complete lines that have been buffered.
        while "\n" in self._buffer.getvalue():
            line, rest = self._buffer.getvalue().split("\n", 1)
            self._buffer = StringIO(rest)
            asyncio.create_task(self._send_line(line + "\n"))
        return len(text)

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
      out.textContent = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => {
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
      ws.onmessage = (ev) => {
        out.textContent += ev.data;
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\n--- Closed ---\n';
      ws.onerror = (e) => out.textContent += '\n--- Error ---\n';
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

        orchestrator = Orchestrator(goal=goal)

        # Capture console output and stream it to the WebSocket client.
        ws_console = WebSocketConsole(websocket)

        # Rich Console writes through a file-like object. With force_terminal=False
        # it strips ANSI escape codes, giving us plain text suitable for the wire.
        from rich.console import Console
        stream_console = Console(file=ws_console, force_terminal=False, width=120)

        # The orchestrator references a module-level ``console``. Swap it so all
        # rich output (including Status objects created with console=console) flows
        # through the WebSocket.
        import furrow.core.orchestrator as orchestrator_module
        original_console = orchestrator_module.console
        orchestrator_module.console = stream_console
        try:
            await orchestrator.run()
        finally:
            orchestrator_module.console = original_console

        await websocket.send_text("\n--- Furrow complete ---\n")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"\n--- Error: {e} ---\n")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)