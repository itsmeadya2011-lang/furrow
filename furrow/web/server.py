from __future__ import annotations

import asyncio
import contextlib
import io
from collections.abc import Iterator
from typing import IO, Any, cast

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

CONSOLE_WIDTH = 120

INDEX_HTML = r"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8" /><title>Furrow</title></head>
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
      out.textContent = '\nStarting...\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      ws.onmessage = (ev) => out.textContent += ev.data + '\n';
      ws.onerror = () => out.textContent += '\nConnection error.\n';
      ws.onclose = () => out.textContent += '\nClosed.\n';
    };
  </script>
</body>
</html>
"""


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class _WebSocketConsoleFile:
    """Write-only file object that queues completed lines for the socket.

    ``rich`` only needs ``write``/``flush``/``isatty``/``encoding`` from a stream,
    so this can stand in for one. ``write`` cannot await, therefore each finished
    line is handed to an ``asyncio.Queue`` that a sender task drains.
    """

    def __init__(self, queue: asyncio.Queue[str | None]) -> None:
        self._queue = queue
        self._buffer = ""
        self.encoding = "utf-8"
        self.errors = "replace"

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._queue.put_nowait(line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            pending, self._buffer = self._buffer, ""
            self._queue.put_nowait(pending.rstrip())

    def emit(self, line: str) -> None:
        self.flush()
        self._queue.put_nowait(line)

    def close(self) -> None:
        self.flush()

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fileno")

    @property
    def closed(self) -> bool:
        return False


async def _pump(queue: asyncio.Queue[str | None], websocket: WebSocket) -> None:
    while True:
        line = await queue.get()
        if line is None:
            return
        try:
            await websocket.send_text(line)
        except Exception:
            return


@contextlib.contextmanager
def _redirect_orchestrator_console(console: Console) -> Iterator[None]:
    """Point the orchestrator module's console at ``console`` for the duration."""
    previous = orchestrator_module.console
    orchestrator_module.console = console
    try:
        yield
    finally:
        orchestrator_module.console = previous


async def _supervise(
    run_task: asyncio.Task[None],
    sink_task: asyncio.Task[None],
    websocket: WebSocket,
) -> str:
    """Watch the run and the socket at once; returns why the handler stops."""
    recv_task: asyncio.Task[Any] = asyncio.create_task(websocket.receive_text())
    try:
        while True:
            done, _ = await asyncio.wait(
                {run_task, sink_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if run_task in done:
                return "run"
            if sink_task in done:
                return "sink"
            error = recv_task.exception()
            if isinstance(error, WebSocketDisconnect):
                return "disconnect"
            if error is not None:
                raise error
            recv_task = asyncio.create_task(websocket.receive_text())
    finally:
        if not recv_task.done():
            recv_task.cancel()
            await asyncio.wait({recv_task})
        if not recv_task.cancelled():
            recv_task.exception()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    sink_task = asyncio.create_task(_pump(queue, websocket))
    run_task: asyncio.Task[None] | None = None
    stream: _WebSocketConsoleFile | None = None
    try:
        try:
            request = StartRequest(**await websocket.receive_json())
        except WebSocketDisconnect:
            return
        except Exception as exc:
            stream = _WebSocketConsoleFile(queue)
            stream.emit(f"Invalid start request: {exc}")
            return

        if request.model:
            from furrow.config import settings

            settings.model = request.model

        stream = _WebSocketConsoleFile(queue)
        console = Console(
            file=cast(IO[str], stream),
            force_terminal=False,
            width=CONSOLE_WIDTH,
        )
        orchestrator = Orchestrator(goal=request.goal)
        orchestrator.console = console  # type: ignore[attr-defined]

        with _redirect_orchestrator_console(console):
            run_task = asyncio.create_task(orchestrator.run())
            try:
                reason = await _supervise(run_task, sink_task, websocket)
            finally:
                run_task.cancel()
                results = await asyncio.gather(run_task, return_exceptions=True)

            failure = next(
                (
                    result
                    for result in results
                    if isinstance(result, BaseException)
                    and not isinstance(result, asyncio.CancelledError)
                ),
                None,
            )
            if failure is not None:
                stream.emit(f"Run failed: {failure}")
            elif reason == "run":
                stream.emit("Run finished.")
    except WebSocketDisconnect:
        pass
    finally:
        if stream is not None:
            stream.close()
        if run_task is not None and not run_task.done():
            run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await queue.put(None)
            await asyncio.gather(sink_task, return_exceptions=True)
        with contextlib.suppress(Exception):
            await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
