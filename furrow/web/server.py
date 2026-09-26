from __future__ import annotations

import asyncio
import io

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


HTML_CONTENT = """<!DOCTYPE html>
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
"""


class _WebSocketFile:
    """File-like object that forwards Console output to a WebSocket connection.

    Rich Console writes to this object synchronously, but WebSocket sends are
    async. We buffer writes and schedule sends on the event loop via flush().
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._buf: list[str] = []

    def write(self, text: str) -> int:
        if text:
            self._buf.append(text)
        return len(text)

    def flush(self) -> None:
        if not self._buf:
            return
        data = "".join(self._buf)
        self._buf.clear()
        if data.strip():
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._ws.send_text(data))
            except RuntimeError:
                pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise io.UnsupportedOperation("not a real file descriptor")


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=HTML_CONTENT)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        ws_file = _WebSocketFile(websocket)
        ws_console = Console(
            file=ws_file,
            force_terminal=False,
            no_color=True,
            width=120,
            highlight=False,
        )

        ws_settings = Settings()
        if model:
            ws_settings.model = model
            ws_settings.planner_model = model
            ws_settings.worker_model = model
            ws_settings.tester_model = model

        orchestrator = Orchestrator(
            goal=goal,
            client=LLMClient(settings=ws_settings),
            console=ws_console,
        )
        await orchestrator.run()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
