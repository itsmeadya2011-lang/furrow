"""FastAPI application that runs one Furrow loop per browser session.

The page, its stylesheet and its script are inline, so the package needs no
static-file directory and no extra packaging metadata.  The interesting part is
the ``/ws`` endpoint: it turns the orchestrator's ``(kind, payload)`` event
stream into JSON frames the page renders as they arrive, validates the first
frame, reports failures to the client instead of dropping the connection
silently, and makes sure a browser that goes away cannot leave an agent loop
running in the background.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from furrow.config import settings
from furrow.core.orchestrator import EventSink, Orchestrator

log = structlog.get_logger(__name__)

# RFC 6455 close codes used by this endpoint.
WS_NORMAL_CLOSURE = 1000
WS_INVALID_PAYLOAD = 1008
WS_INTERNAL_ERROR = 1011

# How long a run may take to unwind after the client asked it to stop before the
# handler cancels it outright.
STOP_GRACE_SECONDS = 5.0


class StartRequest(BaseModel):
    """Payload the client must send as the first websocket frame."""

    goal: str
    model: str | None = None


class StartRequestError(ValueError):
    """Raised when the first websocket frame is not a usable start request."""


_INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Furrow</title>
<style>
  :root { color-scheme: light dark; }
  body {
    font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    margin: 0 auto;
    max-width: 60rem;
    padding: 1.5rem;
  }
  h1 { font-size: 1.2rem; margin: 0 0 1rem; }
  .bar { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
  #goal { flex: 1 1 20rem; padding: 0.4rem; }
  button { padding: 0.4rem 1rem; }
  button[disabled] { opacity: 0.5; }
  #log {
    border: 1px solid #999;
    border-radius: 0.4rem;
    padding: 0.6rem;
    min-height: 14rem;
    max-height: 65vh;
    overflow-y: auto;
    white-space: pre-wrap;
  }
  .cycle { color: #2563eb; font-weight: bold; margin-top: 0.5rem; }
  .ok { color: #16a34a; }
  .bad { color: #dc2626; }
  .warn { color: #d97706; }
  .dim { color: #6b7280; }
</style>
</head>
<body>
<h1>Furrow</h1>
<div class="bar">
  <input id="goal" placeholder="Describe the goal for Furrow" required>
  <button id="start" type="button">Start</button>
  <button id="stop" type="button" disabled>Stop</button>
</div>
<div id="log"></div>
<script>
  const goalEl = document.getElementById('goal');
  const startEl = document.getElementById('start');
  const stopEl = document.getElementById('stop');
  const logEl = document.getElementById('log');
  let socket = null;

  function line(text, cls) {
    const row = document.createElement('div');
    if (cls) row.className = cls;
    row.textContent = text;
    logEl.appendChild(row);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function setRunning(running) {
    goalEl.disabled = running;
    startEl.disabled = running;
    stopEl.disabled = !running;
  }

  function render(ev) {
    const d = ev || {};
    switch (d.type) {
      case 'cycle_start':
        line('== cycle ' + d.cycle + ' ==', 'cycle');
        break;
      case 'plan': {
        const tasks = (d.plan && d.plan.tasks) || [];
        line('plan: ' + tasks.length + ' task(s)', 'dim');
        for (const t of tasks) line('  - ' + t.id + ': ' + t.description, 'dim');
        break;
      }
      case 'task_start':
        line('... ' + d.task_id + ' started: ' + d.description, 'dim');
        break;
      case 'task_completed':
        line('ok  ' + d.task_id + ' completed', 'ok');
        break;
      case 'task_failed':
        line('FAIL ' + d.task_id + ': ' + d.error, 'bad');
        break;
      case 'test_result': {
        const verdict = d.passed ? 'ok  tests passed: ' : 'FAIL tests failed: ';
        line(verdict + d.summary, d.passed ? 'ok' : 'bad');
        for (const f of d.failures || []) line('  - ' + f, 'bad');
        break;
      }
      case 'cycle_end':
        line('cycle ' + d.cycle + (d.done ? ' complete' : ' continuing'), 'dim');
        break;
      case 'halt':
        line('halted: ' + d.reason, 'warn');
        break;
      case 'error':
        line('error: ' + d.message, 'bad');
        break;
      case 'raw':
        line(String(d.message), 'dim');
        break;
      default:
        line('? ' + d.type, 'dim');
    }
  }

  function connect(goal) {
    logEl.textContent = '';
    setRunning(true);
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(scheme + '://' + location.host + '/ws');
    socket = ws;
    // The handshake must finish before send() is legal; sending while the
    // socket is still CONNECTING throws InvalidStateError.
    ws.onopen = function () {
      if (socket !== ws) return;
      ws.send(JSON.stringify({ goal: goal }));
      line('connected to the agent loop', 'dim');
    };
    ws.onmessage = function (event) {
      let parsed = null;
      try {
        parsed = JSON.parse(event.data);
      } catch (err) {
        parsed = { type: 'raw', message: event.data };
      }
      render(parsed);
    };
    ws.onerror = function () {
      line('websocket error', 'bad');
    };
    ws.onclose = function (event) {
      line('disconnected (close code ' + event.code + ')', 'dim');
      if (socket === ws) {
        socket = null;
        setRunning(false);
      }
    };
  }

  startEl.addEventListener('click', function () {
    const goal = goalEl.value.trim();
    if (!goal) {
      line('enter a goal first', 'bad');
      goalEl.focus();
      return;
    }
    connect(goal);
  });

  stopEl.addEventListener('click', function () {
    const ws = socket;
    if (!ws) return;
    line('stop requested', 'warn');
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ stop: true }));
    ws.close();
  });

  goalEl.addEventListener('keydown', function (event) {
    if (event.key === 'Enter' && !goalEl.disabled) startEl.click();
  });
</script>
</body>
</html>
"""


def _log_level(name: str) -> int:
    """Translate a configured log level name into a stdlib logging level."""
    level = getattr(logging, name.strip().upper(), None)
    return level if isinstance(level, int) else logging.INFO


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Configure structlog from settings before the first request is served."""
    level = _log_level(settings.log_level)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s %(message)s")
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(level))
    log.info(
        "furrow web server ready",
        provider=settings.provider.value,
        host=settings.host,
        port=settings.port,
        log_level=logging.getLevelName(level),
    )
    yield


app = FastAPI(title="Furrow", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe for containers and load balancers."""
    return {
        "status": "ok",
        "provider": settings.provider.value,
        "host": settings.host,
        "port": settings.port,
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the self-contained monitoring page."""
    return HTMLResponse(content=_INDEX_HTML)


async def _send_json(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    """Send one JSON frame; return False when the client is already gone."""
    try:
        await websocket.send_json(payload)
    except Exception:  # Starlette raises RuntimeError on a dead socket
        log.debug("dropping %s frame, client is gone", payload.get("type"))
        return False
    return True


async def _close_quietly(websocket: WebSocket, code: int = WS_NORMAL_CLOSURE) -> None:
    """Close the socket, tolerating a peer that disconnected first."""
    try:
        await websocket.close(code=code)
    except Exception:
        log.debug("websocket was already closed")


async def _fail(websocket: WebSocket, message: str, code: int) -> None:
    """Tell the client why the run ended badly, then close with a matching code."""
    await _send_json(websocket, {"type": "error", "message": message})
    await _close_quietly(websocket, code)


def _event_sink(websocket: WebSocket) -> EventSink:
    """Build the ``on_event`` callback that streams orchestrator events to a client.

    The orchestrator also prints to the server's console, so this is purely
    additive; every frame is a JSON object of the form ``{"type": kind, ...data}``.
    A send that fails means the client vanished, which must not raise here: the
    websocket handler owns the teardown, and the orchestrator would otherwise
    keep looping against a dead socket.
    """

    async def sink(kind: str, data: dict[str, Any]) -> None:
        await _send_json(websocket, {"type": kind, **data})

    return sink


def _describe(exc: ValidationError) -> str:
    """Render pydantic validation errors as one short, client-readable line."""
    problems = [
        f"{'.'.join(str(part) for part in error['loc']) or 'payload'}: {error['msg']}"
        for error in exc.errors()
    ]
    return "invalid start request (" + "; ".join(problems) + ")"


def _parse_start(payload: Any) -> StartRequest:
    """Validate a decoded websocket payload as a :class:`StartRequest`."""
    if not isinstance(payload, dict):
        raise StartRequestError(
            f"expected a JSON object with a 'goal', got {type(payload).__name__}"
        )
    try:
        start = StartRequest.model_validate(payload)
    except ValidationError as exc:
        raise StartRequestError(_describe(exc)) from exc
    if not start.goal.strip():
        raise StartRequestError("goal must not be empty")
    return start


async def _receive_start(websocket: WebSocket) -> StartRequest:
    """Read the first frame and validate it as a start request."""
    try:
        payload = await websocket.receive_json()
    except WebSocketDisconnect:
        raise
    except Exception as exc:  # malformed JSON, binary frame, missing 'text'
        raise StartRequestError(f"could not decode the JSON payload: {exc}") from exc
    return _parse_start(payload)


def _apply_model_override(model: str | None) -> None:
    """Point every agent at ``model``, exactly like ``furrow start --model`` does.

    This writes to the process-wide settings singleton, so a model sent by one
    client also applies to later clients; that is acceptable for the default
    loopback bind, which is what the CLI and the UI both default to.
    """
    if not model:
        return
    settings.model = model
    settings.planner_model = model
    settings.worker_model = model
    settings.tester_model = model


def _is_stop_request(message: Mapping[str, Any]) -> bool:
    """True when the client sent ``{"stop": true}``."""
    text = message.get("text")
    if not isinstance(text, str):
        return False
    try:
        payload = json.loads(text)
    except ValueError:
        return False
    return isinstance(payload, dict) and bool(payload.get("stop"))


async def _monitor_client(websocket: WebSocket, on_stop: Callable[[], None]) -> str:
    """Watch the socket until it disconnects or asks the run to stop.

    Returning (rather than raising) is deliberate: the handler uses the reason to
    decide whether to wait for a graceful halt, and a lost socket is a normal
    outcome here, not an error.
    """
    while True:
        try:
            message = await websocket.receive()
        except (RuntimeError, WebSocketDisconnect):
            return "disconnected"
        if message.get("type") == "websocket.disconnect":
            return "disconnected"
        if _is_stop_request(message):
            on_stop()
            return "stop requested"


def _finished_result(task: asyncio.Task[str]) -> str | None:
    """Return a finished task's result, or None when it failed or was cancelled."""
    if not task.done() or task.cancelled() or task.exception() is not None:
        return None
    return task.result()


def _runner_failure(
    runner: asyncio.Task[None],
    done: set[asyncio.Task[Any]],
) -> BaseException | None:
    """Return the exception a finished run died from, if any."""
    if runner not in done or runner.cancelled():
        return None
    return runner.exception()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Run one orchestrator loop per connection, streaming its events to the client."""
    await websocket.accept()
    orchestrator: Orchestrator | None = None
    tasks: list[asyncio.Task[Any]] = []
    try:
        start = await _receive_start(websocket)
        _apply_model_override(start.model)
        orchestrator = Orchestrator(goal=start.goal, on_event=_event_sink(websocket))
        runner = asyncio.create_task(orchestrator.run())
        tasks.append(runner)
        watcher = asyncio.create_task(_monitor_client(websocket, orchestrator.stop))
        tasks.append(watcher)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if runner in pending and _finished_result(watcher) == "stop requested":
            # Give the loop a moment to notice the stop request and emit its halt.
            await asyncio.wait({runner}, timeout=STOP_GRACE_SECONDS)
        failure = _runner_failure(runner, done)
        if failure is None:
            await _close_quietly(websocket, WS_NORMAL_CLOSURE)
        else:
            log.error("orchestrator run failed", error=str(failure))
            await _fail(
                websocket,
                f"{type(failure).__name__}: {failure}",
                code=WS_INTERNAL_ERROR,
            )
    except WebSocketDisconnect:
        log.info("websocket client disconnected before the run started")
    except StartRequestError as exc:
        log.info("rejecting websocket start request: %s", exc)
        await _fail(websocket, str(exc), code=WS_INVALID_PAYLOAD)
    except Exception as exc:
        log.exception("websocket run failed")
        await _fail(
            websocket,
            f"{type(exc).__name__}: {exc}",
            code=WS_INTERNAL_ERROR,
        )
    finally:
        if orchestrator is not None:
            orchestrator.stop()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def run(host: str | None = None, port: int | None = None) -> None:
    """Serve the UI on ``host``/``port``, defaulting to the configured settings.

    The default is loopback on purpose: ``/ws`` starts agents that run project
    commands on this host, so the server must not be reachable from the network
    unless the operator asks for it explicitly.
    """
    uvicorn.run(
        app,
        host=settings.host if host is None else host,
        port=settings.port if port is None else port,
    )