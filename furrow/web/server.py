from __future__ import annotations

import asyncio
import contextlib
import io

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import settings as default_settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class StartResponse(BaseModel):
    goal: str
    status: str
    output: str


class WebSocketStream(io.StringIO):
    def __init__(self, queue: asyncio.Queue[str | None]) -> None:
        super().__init__()
        self._queue = queue

    def write(self, text: str) -> int:
        if text:
            self._queue.put_nowait(text)
        return len(text)


def _make_streaming_console(queue: asyncio.Queue[str | None]) -> Console:
    return Console(
        file=WebSocketStream(queue),
        no_color=True,
        highlight=False,
        width=120,
    )


def _build_orchestrator(goal: str, model: str | None, console: Console) -> Orchestrator:
    settings = default_settings.model_copy(update={"model": model}) if model else None
    return Orchestrator(goal=goal, settings=settings, console=console)


async def _drain_queue(websocket: WebSocket, queue: asyncio.Queue[str | None]) -> None:
    while True:
        msg = await queue.get()
        if msg is None:
            return
        try:
            await websocket.send_text(msg)
        except (WebSocketDisconnect, RuntimeError, OSError):
            return


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <input id="model" placeholder="Optional model override (e.g. gpt-4)" />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = function (e) {
      e.preventDefault();
      out.textContent = 'Starting...\\n';
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(proto + '//' + location.host + '/ws');
      ws.onopen = function () {
        ws.send(JSON.stringify({
          goal: document.getElementById('goal').value,
          model: document.getElementById('model').value || undefined
        }));
      };
      ws.onmessage = function (ev) { out.textContent += ev.data; };
      ws.onclose = function () { out.textContent += '\\n[closed]\\n'; };
      ws.onerror = function () { out.textContent += '\\n[error]\\n'; };
    };
  </script>
</body>
</html>"""
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/start", response_model=StartResponse)
async def start(request: StartRequest) -> StartResponse:
    buffer = io.StringIO()
    console = Console(file=buffer, no_color=True, highlight=False, width=120)
    orchestrator = _build_orchestrator(request.goal, request.model, console)
    await orchestrator.run()
    return StartResponse(goal=request.goal, status="complete", output=buffer.getvalue())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    ws_console = _make_streaming_console(queue)
    drainer = asyncio.create_task(_drain_queue(websocket, queue))
    run_task: asyncio.Task[None] | None = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model") or None
        orchestrator = _build_orchestrator(goal, model, ws_console)
        run_task = asyncio.create_task(orchestrator.run())

        done, _pending = await asyncio.wait(
            {run_task, drainer}, return_when=asyncio.FIRST_COMPLETED
        )

        if run_task in done and not run_task.cancelled():
            exc = run_task.exception()
            if exc is not None:
                queue.put_nowait(f"\nOrchestrator error: {exc}\n")
    except WebSocketDisconnect:
        pass
    finally:
        if run_task is not None and not run_task.done():
            run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await run_task
        await queue.put(None)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(drainer, timeout=5.0)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port, log_level=default_settings.log_level.lower())
