from __future__ import annotations

import asyncio
import html
import ipaddress
import re
from contextlib import suppress
from dataclasses import dataclass
from string import Template

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError
from rich.console import Console

from furrow.config import Settings
from furrow.config import settings as global_settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")
console = Console(stderr=True)

# Seconds the sender gets to flush the queue after the loop has finished.
FLUSH_TIMEOUT = 2.0

# Every model field, so a single override actually reaches the agents.
_MODEL_FIELDS = ("model", "planner_model", "worker_model", "tester_model")

# Hosts that are safe to expose: nothing outside the machine can reach them.
_LOOPBACK_HOSTS = frozenset({"", "localhost", "localhost.localdomain", "127.0.0.1", "::1"})

# Rich console markup such as ``[bold green]`` or ``[/]``. Built from a whitelist of
# style words so ordinary bracketed text (``task[0]``, ``[1, 2]``) is left alone.
_COLOR_NAMES = (
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    "default",
    "bright_black",
    "bright_red",
    "bright_green",
    "bright_yellow",
    "bright_blue",
    "bright_magenta",
    "bright_cyan",
    "bright_white",
)
_STYLE_NAMES = (
    "bold",
    "dim",
    "italic",
    "underline",
    "blink",
    "blink2",
    "reverse",
    "conceal",
    "strike",
    "frame",
    "encircle",
    "overline",
)
_COLOR = "(?:" + "|".join(_COLOR_NAMES) + ")"
_STYLE = "(?:" + "|".join(_STYLE_NAMES) + ")"
# One or more style words in any order, with an optional background.
_TAG_BODY = f"(?:on {_COLOR}|{_COLOR}|{_STYLE})(?:\\s+(?:on {_COLOR}|{_COLOR}|{_STYLE}))*"
_MARKUP_RE = re.compile(rf"\[/?(?:{_TAG_BODY})?\]")
# Colour codes, in case the loop ever renders to a terminal-like console.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@dataclass
class _RunConfig:
    goal: str
    settings: Settings


def strip_markup(text: str) -> str:
    """Turn one emitted loop message into plain text safe to send to a browser.

    The loop emits strings that still carry rich console markup; the browser has
    no way to interpret it, so the tags are removed before they hit the wire.
    A word such as ``[red]`` is unavoidably ambiguous, but real markup is a
    superset of that and is the common case.
    """
    return _MARKUP_RE.sub("", _ANSI_RE.sub("", str(text)))


def _settings_for(model: str | None) -> Settings:
    """Build the settings for one run without touching the global singleton."""
    if not model:
        return global_settings
    # Each agent passes its own per-agent model, so overriding only ``model``
    # would be silently ignored by the planner, workers and tester.
    return global_settings.model_copy(update=dict.fromkeys(_MODEL_FIELDS, model))


def _is_loopback(host: str) -> bool:
    if host.lower() in _LOOPBACK_HOSTS:
        return True
    with suppress(ValueError):
        return ipaddress.ip_address(host).is_loopback
    return False


async def _drain(
    websocket: WebSocket,
    queue: asyncio.Queue[str | None],
    disconnected: asyncio.Event,
) -> None:
    """Forward queued messages to the browser until the sentinel or a failure."""
    try:
        with suppress(Exception):  # a dead socket must not take the run down
            while True:
                message = await queue.get()
                if message is None:
                    return
                await websocket.send_text(message)
    finally:
        disconnected.set()


async def _watch_disconnect(websocket: WebSocket, disconnected: asyncio.Event) -> None:
    """Block until the browser goes away, so the run can be abandoned."""
    try:
        with suppress(Exception):  # a broken receive counts as a disconnect
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return
    finally:
        disconnected.set()


async def _stop(*tasks: asyncio.Task[None]) -> None:
    """Cancel tasks and absorb their results so asyncio never warns about them."""
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _read_start_request(websocket: WebSocket) -> _RunConfig | str:
    """Parse and validate the start payload.

    Returns the run configuration, or a ready-to-send error message. A client
    disconnect propagates to the caller.
    """
    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        raise
    except Exception as exc:
        return f"error: could not read request: {exc}"
    if not isinstance(data, dict):
        return "error: request must be a JSON object."
    try:
        payload = StartRequest(**data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "body"
        return f"error: invalid request: {where}: {first['msg']}"
    goal = payload.goal.strip()
    if not goal:
        return "error: goal must not be empty."
    return _RunConfig(goal=goal, settings=_settings_for(payload.model))


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/")
async def index(goal: str | None = None) -> HTMLResponse:
    """The single-page monitor; ``?goal=`` pre-fills the form."""
    return HTMLResponse(
        content=Template(_PAGE).safe_substitute(goal=html.escape(goal or "", quote=True))
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    disconnected = asyncio.Event()

    def push(message: str) -> None:
        """Queue one loop message. Never raises back into the loop."""
        with suppress(Exception):
            queue.put_nowait(strip_markup(message))

    async def report(message: str) -> None:
        with suppress(Exception):
            await websocket.send_text(strip_markup(message))

    sender = asyncio.create_task(_drain(websocket, queue, disconnected))
    producers: list[asyncio.Task[None]] = []
    try:
        config = await _read_start_request(websocket)
        if isinstance(config, str):
            await report(config)
            return
        run_task = asyncio.create_task(
            Orchestrator(goal=config.goal, client=LLMClient(config.settings), on_event=push).run()
        )
        watcher = asyncio.create_task(_watch_disconnect(websocket, disconnected))
        producers = [run_task, watcher]
        # The loop and the socket are driven concurrently: the loop only fills
        # the queue, this handler keeps sending until one side stops.
        await asyncio.wait({run_task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if run_task.done() and not run_task.cancelled():
            failure = run_task.exception()
            if failure is not None:
                await report(f"error: run failed: {failure}")
        else:
            # The browser went away first: stop burning tokens on nobody.
            run_task.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        await _stop(*producers)
        queue.put_nowait(None)  # let the sender drain and exit
        await asyncio.wait({sender}, timeout=FLUSH_TIMEOUT)
        with suppress(Exception):
            await websocket.close()
        await _stop(sender)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Serve the monitor with uvicorn."""
    if not _is_loopback(host):
        console.print(
            f"[yellow]Warning: binding to {host} exposes an unauthenticated endpoint"
            " that can run code on this machine. Use 127.0.0.1 unless you mean it."
            "[/yellow]"
        )
    uvicorn.run(app, host=host, port=port)


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Furrow</title>
<style>
  body { font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 54rem; }
  h1 { font-size: 1.2rem; }
  form { display: flex; gap: .5rem; flex-wrap: wrap; }
  input { flex: 1 1 18rem; padding: .45rem; }
  #model { flex: 0 1 14rem; }
  button { padding: .45rem 1.1rem; }
  button[disabled] { opacity: .5; cursor: progress; }
  #out { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .85rem;
         white-space: pre-wrap; word-break: break-word; background: rgba(128,128,128,.14);
         border-radius: .3rem; padding: .75rem; min-height: 10rem; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required value="$goal" />
    <input id="model" placeholder="model (optional)" />
    <button id="start" type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const start = document.getElementById('start');
    const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
    form.onsubmit = (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value.trim();
      if (!goal) { out.textContent = 'A goal is required.\\n'; return; }
      const model = document.getElementById('model').value.trim();
      const ws = new WebSocket(wsUrl);
      out.textContent = '';
      start.disabled = true;
      ws.onopen = () => ws.send(JSON.stringify(model ? {goal: goal, model: model} : {goal: goal}));
      ws.onmessage = (ev) => { out.textContent += ev.data + '\\n'; };
      ws.onerror = () => { out.textContent += 'WebSocket error.\\n'; };
      ws.onclose = () => {
        start.disabled = false;
        out.textContent += '\\n--- Done ---\\n';
      };
    };
  </script>
</body>
</html>
"""
