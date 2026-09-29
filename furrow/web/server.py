from __future__ import annotations

import asyncio
import re

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    provider: str | None = None
    max_cycles: int | None = None
    max_parallel: int | None = None


@app.get("/")
async def index() -> HTMLResponse:
    html = """<!DOCTYPE html>
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
      out.textContent += 'Starting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


class _WebSocketFile:
    """File-like object that routes console output lines to a WebSocket."""

    _ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            cleaned = self._ansi_re.sub("", line).strip()
            if cleaned:
                self._schedule_send(cleaned)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            cleaned = self._ansi_re.sub("", self._buf).strip()
            if cleaned:
                self._schedule_send(cleaned)
            self._buf = ""

    def _schedule_send(self, text: str) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon(self._do_send, text)
        except RuntimeError:
            pass

    def _do_send(self, text: str) -> None:
        asyncio.ensure_future(self._ws.send_text(text))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        # Build settings with optional overrides from the request.
        settings = Settings()
        model = data.get("model")
        if model:
            settings.model = model
            settings.planner_model = model
            settings.worker_model = model
            settings.tester_model = model
        provider = data.get("provider")
        if provider:
            settings.provider = Provider(provider)
        max_cycles = data.get("max_cycles")
        if max_cycles is not None:
            settings.max_cycles = max_cycles
        max_parallel = data.get("max_parallel")
        if max_parallel is not None:
            settings.max_parallel_tasks = max_parallel

        ws_console = Console(
            file=_WebSocketFile(websocket),  # type: ignore[arg-type]
            force_terminal=False,
            no_color=True,
            width=120,
        )
        orchestrator = Orchestrator(
            goal=goal, settings=settings, console=ws_console
        )
        await orchestrator.run()
        await websocket.send_text("\n[Done]")
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001 - send errors back to client over websocket
        await websocket.send_text(f"\n[Error: {e}]")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
