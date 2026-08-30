from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketWriter:
    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    def write(self, text: str) -> int:
        cleaned = _ANSI_ESCAPE.sub("", text)
        if cleaned:
            asyncio.create_task(self._websocket.send_text(cleaned))
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
    <input id="model" placeholder="Model (optional)" />
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
      ws.send(JSON.stringify({goal: document.getElementById('goal').value, model: document.getElementById('model').value || null}));
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

        settings = Settings(workspace=Path.cwd())
        if model:
            settings.model = model

        stop_event = asyncio.Event()
        ws_writer = WebSocketWriter(websocket)
        console = Console(file=ws_writer, no_color=True)
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal=goal, client=client, console=console, stop_event=stop_event)

        run_task = asyncio.create_task(orchestrator.run())
        disconnect_task = asyncio.create_task(_wait_for_disconnect(websocket))

        done, pending = await asyncio.wait(
            [run_task, disconnect_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if disconnect_task in done:
            stop_event.set()

        for task in pending:
            task.cancel()

        await asyncio.gather(run_task, disconnect_task, return_exceptions=True)
    except WebSocketDisconnect:
        pass


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    while True:
        try:
            await websocket.receive_text()
        except WebSocketDisconnect:
            return


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
