"""FastAPI front-end for Furrow.

The ``/ws`` endpoint runs an :class:`Orchestrator` while streaming its Rich
output back to the browser as JSON frames.  The run is cancellable from the
client and bounded by :data:`MAX_RUN_SECONDS` so a single websocket can never
hold the process forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import re
import sys
from collections.abc import Iterator
from typing import IO, Any, cast

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError
from rich.console import Console

from furrow.config import settings
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

#: Hard ceiling for a single websocket run.  Enforced regardless of settings.
MAX_RUN_SECONDS = 1800

#: Matches the orchestrator's ``═══ Cycle N ═══`` banners.
_CYCLE_RE = re.compile(r"Cycle\s+(\d+)")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

#: Internal queue markers; never forwarded to the client.
_CANCEL = "__cancel__"
_DISCONNECT = "__disconnect__"


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


def _accepts_console_kwarg() -> bool:
    """True when ``Orchestrator.__init__`` takes an explicit ``console``."""
    try:
        signature = inspect.signature(Orchestrator.__init__)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    return "console" in signature.parameters


_ACCEPTS_CONSOLE_KWARG = _accepts_console_kwarg()


class _OutputState:
    """Turns captured Rich text into queued websocket frames."""

    def __init__(self, frames: asyncio.Queue) -> None:
        self.frames = frames
        self.cycle = 0

    def emit(self, text: str) -> None:
        """Queue ``text`` as a log frame, flagging cycle banners."""
        match = _CYCLE_RE.search(text)
        if match is not None:
            cycle = int(match.group(1))
            if cycle != self.cycle:
                self.cycle = cycle
                self.frames.put_nowait({"type": "status", "cycle": cycle})
        self.frames.put_nowait({"type": "log", "text": text})


class _OutputTap(io.TextIOBase):
    """Write-only tee: mirrors Rich output to stdout and into the queue.

    Using a tap as the console's ``file`` means *everything* the console emits
    (``print``, ``print_json``, ``rule``, panels, ...) is captured, so no
    specific rendering call can silently bypass the stream.

    ``io.TextIOBase`` is the base ``rich.console.Console(file=...)`` expects for
    a text stream; it also gives us the standard no-op/unsupported-operation
    fallbacks for every method the console never calls.
    """

    def __init__(self, state: _OutputState, stream: Any) -> None:
        self._state = state
        self._stream = stream

    def write(self, text: str) -> int:
        with contextlib.suppress(Exception):
            self._stream.write(text)
        for line in _ANSI_RE.sub("", text).splitlines():
            line = line.rstrip()
            if line.strip():
                self._state.emit(line)
        return len(text)

    def flush(self) -> None:
        with contextlib.suppress(Exception):
            self._stream.flush()


@contextlib.contextmanager
def _console_scope(console: Console) -> Iterator[None]:
    """Make ``console`` the console the orchestrator renders to.

    Preferred path is the ``Orchestrator(console=...)`` keyword.  When the
    orchestrator does not accept it, the module-level global it closes over is
    swapped for the duration of the run and restored afterwards.
    """
    if _ACCEPTS_CONSOLE_KWARG or not hasattr(orchestrator_module, "console"):
        yield
        return
    previous = orchestrator_module.console
    orchestrator_module.console = console
    try:
        yield
    finally:
        orchestrator_module.console = previous


def _build_orchestrator(goal: str, console: Console) -> Orchestrator:
    """Construct the orchestrator through its stable public surface."""
    kwargs: dict[str, Any] = {"goal": goal}
    if _ACCEPTS_CONSOLE_KWARG:
        kwargs["console"] = console
    return Orchestrator(**kwargs)


async def _drive(request: StartRequest, frames: asyncio.Queue) -> str | None:
    """Run the orchestrator, streaming frames into ``frames``.

    Returns ``None`` on a clean finish, or a message describing the failure.
    """
    state = _OutputState(frames)
    tap = _OutputTap(state, sys.stdout)
    # Cast bridges a typeshed gap: ``io.TextIOBase`` is not a subclass of
    # ``typing.IO[str]``, though it is a valid runtime text stream for Rich.
    console = Console(file=cast("IO[str]", tap), soft_wrap=True)
    model = request.model or settings.model
    frames.put_nowait(
        {
            "type": "log",
            "text": (
                f"starting run: provider={settings.provider.value} model={model} "
                f"max_cycles={settings.max_cycles} cap={MAX_RUN_SECONDS}s"
            ),
        }
    )
    try:
        with _console_scope(console):
            orchestrator = _build_orchestrator(request.goal, console)
            await orchestrator.run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - any failure must reach the client as `done`
        return f"{type(exc).__name__}: {exc}"
    return None


async def _read_inbound(websocket: WebSocket, frames: asyncio.Queue) -> None:
    """Watch for ``{"type": "cancel"}`` until the socket closes."""
    try:
        while True:
            message = await websocket.receive_json()
            if isinstance(message, dict) and message.get("type") == "cancel":
                frames.put_nowait({"type": _CANCEL})
                return
    except (WebSocketDisconnect, RuntimeError, ValueError):
        frames.put_nowait({"type": _DISCONNECT})


async def _send(websocket: WebSocket, frame: dict) -> bool:
    """Send one frame; ``False`` if the connection is already gone."""
    try:
        await websocket.send_json(frame)
    except (WebSocketDisconnect, RuntimeError, OSError):
        return False
    return True


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(BaseException):
        await task


def _settle_status(task: asyncio.Task) -> tuple[str, str]:
    """Classify a finished run task as (status, reason)."""
    if task.cancelled():
        return "stopped", "run cancelled"
    if task.exception() is not None:
        exc = task.exception()
        assert exc is not None
        return "error", f"{type(exc).__name__}: {exc}"
    note = task.result()
    if note:
        return "error", str(note)
    return "completed", "run finished"


async def _supervise(
    websocket: WebSocket, frames: asyncio.Queue, run_task: asyncio.Task
) -> tuple[str, str]:
    """Pump queued frames to the socket until the run settles."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + MAX_RUN_SECONDS
    while True:
        if run_task.done():
            return _settle_status(run_task)
        remaining = deadline - loop.time()
        if remaining <= 0:
            await _cancel(run_task)
            return "stopped", f"run exceeded the {MAX_RUN_SECONDS}s safety limit and was cancelled"
        try:
            frame = await asyncio.wait_for(frames.get(), remaining)
        except asyncio.TimeoutError:
            continue
        kind = str(frame.get("type", ""))
        if kind == _CANCEL:
            await _cancel(run_task)
            return "stopped", "cancelled by client request"
        if kind == _DISCONNECT:
            await _cancel(run_task)
            return "stopped", "client disconnected"
        if not await _send(websocket, frame):
            await _cancel(run_task)
            return "stopped", "client connection lost"


async def _drain(websocket: WebSocket, frames: asyncio.Queue) -> None:
    """Flush output queued just before the run ended."""
    while not frames.empty():
        frame = frames.get_nowait()
        if str(frame.get("type", "")).startswith("__"):
            continue
        if not await _send(websocket, frame):
            return


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow &middot; agent console</title>
<style>
  body {
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    line-height: 1.45;
    max-width: 58rem;
    margin: 0 auto;
    padding: 1.5rem;
  }
  h1 { font-size: 1.3rem; margin: 0 0 1rem; }
  label { display: block; font-weight: 600; margin-bottom: 0.35rem; }
  textarea {
    width: 100%;
    box-sizing: border-box;
    font: inherit;
    padding: 0.5rem;
  }
  .row { display: flex; gap: 0.5rem; margin-top: 0.75rem; }
  button { font: inherit; padding: 0.45rem 0.9rem; cursor: pointer; }
  button[disabled] { cursor: not-allowed; opacity: 0.55; }
  #out {
    margin-top: 1rem;
    padding: 0.75rem;
    height: 26rem;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
    background: rgba(127, 127, 127, 0.12);
    border-radius: 0.5rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.85rem;
  }
  .status { color: #2f6feb; font-weight: 700; }
  .error { color: #c0392b; font-weight: 700; }
  .done { color: #1e8449; font-weight: 700; }
</style>
</head>
<body>
<h1>Furrow</h1>
<form id="form">
  <label for="goal">Goal</label>
  <textarea id="goal" name="goal" rows="3" required
    placeholder="Describe what the agent should build"></textarea>
  <div class="row">
    <button id="start" type="submit">Start</button>
    <button id="stop" type="button" disabled>Stop</button>
  </div>
</form>
<pre id="out" aria-live="polite"></pre>
<script>
  const MAX_LINES = 500;
  const form = document.getElementById('form');
  const goalEl = document.getElementById('goal');
  const startBtn = document.getElementById('start');
  const stopBtn = document.getElementById('stop');
  const out = document.getElementById('out');
  let ws = null;
  let lines = [];

  function render() {
    out.textContent = lines.join('\\n');
    out.scrollTop = out.scrollHeight;
  }

  function append(text, cls) {
    lines.push(cls ? '[' + cls + '] ' + text : String(text));
    if (lines.length > MAX_LINES) {
      lines = lines.slice(lines.length - MAX_LINES);
    }
    render();
  }

  function setRunning(running) {
    startBtn.disabled = running;
    stopBtn.disabled = !running;
    goalEl.disabled = running;
  }

  function finish(text) {
    append(text, 'done');
    setRunning(false);
    if (ws) {
      ws.onclose = null;
      ws.close();
      ws = null;
    }
  }

  form.onsubmit = (ev) => {
    ev.preventDefault();
    if (ws) { return; }
    const goal = goalEl.value.trim();
    if (!goal) { append('a goal is required', 'error'); return; }
    lines = [];
    render();
    setRunning(true);
    append('connecting...');
    const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
    ws = new WebSocket(scheme + location.host + '/ws');
    ws.onopen = () => { ws.send(JSON.stringify({ goal: goal, model: null })); };
    ws.onmessage = (message) => {
      let frame;
      try {
        frame = JSON.parse(message.data);
      } catch (err) {
        append(message.data, 'error');
        return;
      }
      if (frame.type === 'log') { append(frame.text, ''); return; }
      if (frame.type === 'status') { append('cycle ' + frame.cycle, 'status'); return; }
      if (frame.type === 'error') { finish(frame.text); return; }
      if (frame.type === 'done') { finish(frame.status + ': ' + frame.reason); return; }
      append(JSON.stringify(frame), '');
    };
    ws.onerror = () => {
      append('websocket error', 'error');
      setRunning(false);
    };
    ws.onclose = () => {
      if (ws) { append('connection closed', 'error'); ws = null; setRunning(false); }
    };
  };

  stopBtn.onclick = () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'cancel' }));
    }
  };
</script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    frames: asyncio.Queue = asyncio.Queue()
    run_task: asyncio.Task | None = None
    reader: asyncio.Task | None = None
    try:
        try:
            payload = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        except (RuntimeError, ValueError) as exc:
            await _send(websocket, {"type": "error", "text": f"invalid JSON payload: {exc}"})
            return
        try:
            request = StartRequest.model_validate(payload)
        except ValidationError as exc:
            await _send(
                websocket,
                {"type": "error", "text": f"invalid request: {_validation_message(exc)}"},
            )
            return
        if not request.goal.strip():
            await _send(websocket, {"type": "error", "text": "goal must not be blank"})
            return

        run_task = asyncio.create_task(_drive(request, frames))
        reader = asyncio.create_task(_read_inbound(websocket, frames))
        status, reason = await _supervise(websocket, frames, run_task)
        await _drain(websocket, frames)
        await _send(websocket, {"type": "done", "status": status, "reason": reason})
    except WebSocketDisconnect:
        if run_task is not None:
            run_task.cancel()
    finally:
        for task in (run_task, reader):
            if task is not None and not task.done():
                task.cancel()
        for task in (run_task, reader):
            if task is not None:
                with contextlib.suppress(BaseException):
                    await task
        with contextlib.suppress(Exception):
            await websocket.close()


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors()[:3]:
        location = ".".join(str(item) for item in error.get("loc", ())) or "payload"
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return "; ".join(parts) or str(exc)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)