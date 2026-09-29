from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

log = logging.getLogger(__name__)

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None
    provider: Optional[str] = None
    max_cycles: Optional[int] = None
    max_parallel: Optional[int] = None


# Track active orchestrations for status/stop
_active_sessions: dict[str, Orchestrator] = {}


class _WebSocketOutput:
    """A minimal file-like object that streams text to a WebSocket.

    Rich Console calls ``write`` synchronously, but we need to send to an
    async WebSocket. We buffer chunks and schedule async sends on the loop.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._buffer: list[str] = []
        self._flush_task: Optional[asyncio.Task] = None

    def write(self, data: str) -> int:
        if data.strip():
            self._buffer.append(data)
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._flush())
        return len(data)

    def flush(self) -> None:
        if self._buffer and (self._flush_task is None or self._flush_task.done()):
            self._flush_task = asyncio.create_task(self._flush())

    async def _flush(self) -> None:
        while self._buffer:
            chunk = self._buffer.pop(0)
            try:
                await self._ws.send_text(chunk)
            except (WebSocketDisconnect, Exception):
                self._buffer.clear()
                break


def _settings_from_request(req: StartRequest) -> Settings:
    kwargs: dict = {}
    if req.model:
        kwargs["model"] = req.model
    if req.provider:
        kwargs["provider"] = req.provider
    if req.max_cycles is not None:
        kwargs["max_cycles"] = req.max_cycles
    if req.max_parallel is not None:
        kwargs["max_parallel_tasks"] = req.max_parallel
    return Settings(**kwargs) if kwargs else Settings()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
  #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 6px; font-family: monospace; font-size: 14px; white-space: pre-wrap; height: 500px; overflow-y: auto; }
  input { width: 80%; padding: 0.5rem; }
  button { padding: 0.5rem 1rem; }
</style>
</head>
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
    let ws;
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => { out.textContent += ev.data + '\\n'; out.scrollTop = out.scrollHeight; };
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
    session_id: Optional[str] = None
    ws_output: Optional[_WebSocketOutput] = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal:
            await websocket.send_text("Error: no goal provided")
            await websocket.close(code=1003)
            return

        req = StartRequest(**{
            k: data.get(k) for k in ("model", "provider", "max_cycles", "max_parallel")
        })
        settings = _settings_from_request(req)

        ws_output = _WebSocketOutput(websocket)
        ws_console = Console(
            file=ws_output,
            force_terminal=False,
            no_color=True,
            width=120,
            highlight=False,
        )
        orchestrator = Orchestrator(goal=goal, settings=settings, console=ws_console)
        session_id = id(orchestrator)
        _active_sessions[session_id] = orchestrator

        await orchestrator.run()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.exception("WebSocket error")
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception:
            pass
    finally:
        if session_id and session_id in _active_sessions:
            del _active_sessions[session_id]
        if ws_output is not None:
            ws_output.flush()
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "active_sessions": len(_active_sessions)}


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    logging.basicConfig(level="INFO")
    uvicorn.run(app, host=host, port=port)
