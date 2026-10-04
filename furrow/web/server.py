"""FastAPI + WebSocket front end for Furrow.

A browser tab opens a WebSocket to ``/ws``, sends a single
``{"goal": ..., "model": ...}`` message, and then receives the orchestrator's
progress events (``loop_started``, ``cycle_started``, ``plan_created``,
``task_started``, ``task_completed``, ``task_failed``, ``tests_completed``,
``cycle_failed``, ``stop_requested``, ``loop_stopped``) as JSON frames while its
run is in flight.

Design notes:

* Every connection owns its own :class:`~furrow.core.orchestrator.Orchestrator`,
  its own :class:`~furrow.llm.LLMClient` and its own run task, so two tabs
  cannot corrupt one another. The process-wide ``furrow.config.settings``
  singleton is only ever read: a ``model`` override is applied to a
  per-connection copy, never to the global.
* Sends to the socket are serialised and guarded, so a broken or vanished
  client can never kill the agent loop or the run task.
* Closing the tab or pressing *Stop* calls ``Orchestrator.stop()`` and then
  cancels the run task if it has not wound down within
  :data:`STOP_GRACE_SECONDS`, because an unwatched run still spends the user's
  API key.
* Nothing is ever swallowed silently: bad input, missing API keys and
  unexpected failures all reach the browser as a structured ``error`` frame
  before the socket closes.
* The endpoint has no authentication and accepts arbitrary goals, so
  :func:`run` binds loopback unless the caller explicitly asks otherwise (and
  shouts about it when they do).
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError
from rich.console import Console

from furrow.config import Settings, get_settings
from furrow.core.orchestrator import EventCallback, Orchestrator
from furrow.llm import LLMClient, LLMConfigError, LLMError, LLMRateLimitError

__all__ = ["MAX_CONCURRENT_RUNS", "STOP_GRACE_SECONDS", "StartRequest", "app", "run"]

logger = logging.getLogger(__name__)

#: How long a stopping run may take to wind down before its task is cancelled.
STOP_GRACE_SECONDS = 15.0
#: How many runs may be in flight at once. Enough for a couple of tabs, small
#: enough that one runaway loop cannot spend the whole API key budget.
MAX_CONCURRENT_RUNS = 4

app = FastAPI(title="Furrow")

_console = Console(stderr=True, soft_wrap=True)


class StartRequest(BaseModel):
    """The one message a client sends after the WebSocket is accepted."""

    goal: str = ""
    model: str | None = None


class _BadRequest(Exception):
    """A start message the client got wrong; the message was already reported."""


def _log(message: str) -> None:
    """Record a diagnostic. Logging must never be able to break a run."""
    logger.info(message)


# -- per-connection plumbing ------------------------------------------------


@dataclass
class _Session:
    """Everything one WebSocket connection owns.

    A session is never shared, so two tabs cannot interleave frames onto the
    same socket or step on each other's orchestrator state.
    """

    websocket: WebSocket
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    open: bool = True

    async def send(self, payload: dict[str, Any]) -> bool:
        """Send one JSON frame. Returns False if the client is gone."""
        if not self.open:
            return False
        async with self.lock:
            if not self.open:
                return False
            try:
                await self.websocket.send_json(payload)
            except Exception as exc:  # noqa: BLE001 - a failed UI write is not fatal
                self.open = False
                _log(f"websocket send failed: {type(exc).__name__}: {exc}")
                return False
        return True

    async def close(self) -> None:
        """Close the socket, tolerating a client that already went away."""
        async with self.lock:
            if not self.open:
                return
            self.open = False
            try:
                await self.websocket.close()
            except Exception as exc:  # noqa: BLE001 - nothing useful left to do
                _log(f"websocket close failed: {type(exc).__name__}: {exc}")


_slot_lock = threading.Lock()
_active_runs = 0


def _acquire_slot() -> bool:
    """Reserve one of the :data:`MAX_CONCURRENT_RUNS` run slots."""
    global _active_runs
    with _slot_lock:
        if _active_runs >= MAX_CONCURRENT_RUNS:
            return False
        _active_runs += 1
        return True


def _release_slot() -> None:
    """Give a run slot back."""
    global _active_runs
    with _slot_lock:
        _active_runs = max(0, _active_runs - 1)


# -- payload helpers --------------------------------------------------------


def _error_payload(message: str, kind: str, cycle: int = 0) -> dict[str, Any]:
    """Build the structured frame the browser renders for any failure."""
    return {
        "event": "error",
        "cycle": cycle,
        "kind": kind,
        "error": message,
        "fatal": True,
    }


def _classify(exc: BaseException) -> tuple[str, str]:
    """Map an exception to a ``(kind, message)`` pair meant for a human."""
    detail = str(exc).strip() or type(exc).__name__
    if isinstance(exc, LLMConfigError):
        return (
            "config",
            f"{detail} - set the API key for your provider (ANTHROPIC_API_KEY / "
            "OPENAI_API_KEY, or the FURROW_* equivalents) and try again.",
        )
    if isinstance(exc, LLMRateLimitError):
        return "rate_limit", f"{detail} - the provider rate limited this run."
    if isinstance(exc, LLMError):
        return "llm", f"{detail} - the language model request failed."
    if isinstance(exc, ValueError):
        return "config", f"{detail} - check the Furrow settings."
    return "error", f"{detail} (unexpected {type(exc).__name__})."


def _parse_start(raw: Any) -> StartRequest:
    """Validate a start message.

    Raises ``ValueError`` (including ``ValidationError``) with a message that is
    meant to be shown to whoever sent it.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            'expected a JSON object such as {"goal": "write the tests"}, got '
            f"{type(raw).__name__}"
        )
    request = StartRequest.model_validate(raw)
    request.goal = request.goal.strip()
    request.model = (request.model or "").strip() or None
    if not request.goal:
        raise ValueError("'goal' is required and must not be blank")
    return request


def _settings_for(model: str | None) -> Settings:
    """Settings for one connection, with an optional per-connection override.

    ``furrow.config.settings`` is a process-wide singleton, so the override is
    applied to a copy instead of being mutated in place; otherwise the model
    picked in one tab would silently become the model every other tab uses.
    Mirrors the CLI's ``--model``: the override also covers the per-role model
    settings, which is what the agents actually call with.
    """
    base = get_settings()
    if not model:
        return base
    return base.model_copy(
        update={
            "model": model,
            "planner_model": model,
            "worker_model": model,
            "tester_model": model,
        }
    )


def _event_sink(session: _Session) -> EventCallback:
    """Build the orchestrator event callback that streams to one socket."""

    async def send_event(payload: dict[str, Any]) -> None:
        await session.send(payload)

    return send_event


def _task_error(task: asyncio.Task[None]) -> BaseException | None:
    """The exception a finished run task died of, if any."""
    if task.cancelled():
        return None
    return task.exception()


async def _stop_run(
    session: _Session,
    orchestrator: Orchestrator,
    run_task: asyncio.Task[None],
) -> str:
    """Ask the loop to stop, then cancel the run task if it lingers."""
    await session.send({"event": "stopping", "cycle": orchestrator.cycles})
    try:
        await orchestrator.stop()
    except Exception as exc:  # noqa: BLE001 - stopping must not raise
        _log(f"Orchestrator.stop() failed: {type(exc).__name__}: {exc}")
    if run_task.done():
        return "stop requested"
    timed_out = False
    try:
        await asyncio.wait_for(asyncio.shield(run_task), timeout=STOP_GRACE_SECONDS)
    except asyncio.TimeoutError:
        timed_out = True
    except asyncio.CancelledError:
        run_task.cancel()
        raise
    except Exception as exc:  # noqa: BLE001 - the run's own error is reported later
        _log(f"run raised while stopping: {type(exc).__name__}: {exc}")
    if not timed_out:
        return "stopped"
    _log(f"run did not stop within {STOP_GRACE_SECONDS:g}s; cancelling it")
    run_task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await run_task
    return f"cancelled after ignoring the stop for {STOP_GRACE_SECONDS:g}s"


async def _watch_client(session: _Session) -> str:
    """Wait for a stop request or a disconnect; report why the wait ended."""
    while True:
        try:
            # receive() rather than receive_text() so a stray binary frame
            # cannot turn into a protocol error that looks like a stop.
            message = await session.websocket.receive()
        except WebSocketDisconnect:
            return "client disconnected"
        except RuntimeError:
            # Already closed on this side; treat it as the client going away.
            return "socket closed"
        if message["type"] == "websocket.disconnect":
            return "client disconnected"
        raw = message.get("text")
        if not isinstance(raw, str):
            raw = (message.get("bytes") or b"").decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            continue  # ignore junk; the run is what matters
        if isinstance(payload, dict) and str(payload.get("action", "")).lower() == "stop":
            return "stop requested"


async def _receive_start(session: _Session) -> StartRequest:
    """Read and validate the single start message a client sends.

    Raises ``_BadRequest`` once the client has been told what was wrong, and
    lets ``WebSocketDisconnect`` through for the normal closed-tab path.
    """
    try:
        raw = await session.websocket.receive_json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        message = f"start message was not valid JSON: {exc}"
        await session.send(_error_payload(message, "invalid_json"))
        raise _BadRequest(message) from exc
    try:
        return _parse_start(raw)
    except (ValidationError, ValueError, TypeError) as exc:
        message = f"invalid start message: {exc}"
        await session.send(_error_payload(message, "invalid_request"))
        raise _BadRequest(message) from exc


async def _run(session: _Session, request: StartRequest) -> None:
    """Drive one orchestrator run, streaming events until it ends."""
    if not _acquire_slot():
        await session.send(
            _error_payload(
                f"the server is already running {MAX_CONCURRENT_RUNS} run(s); "
                "stop one of them before starting another",
                "busy",
            )
        )
        return

    try:
        client = LLMClient(settings=_settings_for(request.model))
        orchestrator = Orchestrator(
            goal=request.goal,
            client=client,
            console=Console(stderr=True, soft_wrap=True),
            on_event=_event_sink(session),
        )
    except Exception as exc:  # noqa: BLE001 - report, never die quietly
        _release_slot()
        kind, message = _classify(exc)
        _log(f"could not start the run: {message}")
        await session.send(_error_payload(message, kind))
        return

    run_task = asyncio.create_task(orchestrator.run(), name="furrow-run")
    control_task = asyncio.create_task(_watch_client(session), name="furrow-control")
    outcome = "finished"
    try:
        # The run is the long coroutine; the control task reports a stop or a
        # disconnect while it is still going.
        await asyncio.wait({run_task, control_task}, return_when=asyncio.FIRST_COMPLETED)
        if not run_task.done():
            outcome = await _stop_run(session, orchestrator, run_task)
        error = _task_error(run_task)
        if error is not None:
            kind, message = _classify(error)
            _log(f"run failed: {message}")
            await session.send(_error_payload(message, kind, cycle=orchestrator.cycles))
            outcome = f"failed: {message}"
        await session.send(
            {"event": "run_finished", "cycle": orchestrator.cycles, "reason": outcome}
        )
    except asyncio.CancelledError:
        run_task.cancel()
        raise
    finally:
        try:
            control_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await control_task
            # The orchestrator only closes a client it created itself.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await client.aclose()
        finally:
            _release_slot()


# -- routes -----------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Run one goal per connection, streaming every orchestrator event back."""
    await websocket.accept()
    session = _Session(websocket=websocket)
    try:
        request = await _receive_start(session)
    except WebSocketDisconnect:
        _log("client disconnected before sending a goal")
        return
    except _BadRequest:
        await session.close()
        return
    except Exception as exc:  # noqa: BLE001 - report, never die quietly
        kind, message = _classify(exc)
        _log(f"could not read the start message: {message}")
        await session.send(_error_payload(message, kind))
        await session.close()
        return

    await _run(session, request)
    await session.close()


@app.get("/")
async def index() -> HTMLResponse:
    """Serve the single-page UI."""
    return HTMLResponse(content=_INDEX_HTML)


#: The whole UI in one self-contained page: no build step, no external assets.
#:
#: This is a *raw* string on purpose. The JavaScript contains ``\n`` escapes
#: that must reach the browser as the two characters backslash + n; in a
#: non-raw Python string they would become real newlines and break the JS
#: string literals they sit in.
_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow</title>
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    margin: 0 auto; max-width: 64rem; padding: 1.5rem; line-height: 1.45;
  }
  h1 { font-size: 1.35rem; margin: 0 0 0.75rem; }
  form { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem; }
  input { flex: 1 1 16rem; padding: 0.45rem 0.55rem; font: inherit; }
  button { padding: 0.45rem 0.9rem; font: inherit; cursor: pointer; }
  button[disabled] { cursor: not-allowed; opacity: 0.55; }
  #status { margin: 0 0 0.75rem; font-weight: 600; }
  #events {
    list-style: none; margin: 0; padding: 0; border: 1px solid #8885;
    max-height: 65vh; overflow: auto;
  }
  #events li {
    padding: 0.3rem 0.5rem; border-bottom: 1px solid #8883;
    white-space: pre-wrap; overflow-wrap: anywhere;
  }
  #events li:last-child { border-bottom: none; }
  .tag { opacity: 0.7; margin-right: 0.5rem; }
  .info { background: rgba(56, 139, 253, 0.12); }
  .ok { background: rgba(46, 160, 67, 0.16); }
  .warn { background: rgba(210, 153, 34, 0.18); }
  .bad { background: rgba(248, 81, 73, 0.18); }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required autocomplete="off" />
    <input id="model" placeholder="Model override (optional)" autocomplete="off" />
    <button id="start" type="submit">Start</button>
    <button id="stop" type="button" disabled>Stop</button>
  </form>
  <p id="status">Idle.</p>
  <ul id="events"></ul>
<script>
  (function () {
    const form = document.getElementById('form');
    const goalInput = document.getElementById('goal');
    const modelInput = document.getElementById('model');
    const startBtn = document.getElementById('start');
    const stopBtn = document.getElementById('stop');
    const statusEl = document.getElementById('status');
    const listEl = document.getElementById('events');
    let socket = null;
    let failed = false;

    function text(value) {
      if (typeof value === 'string') { return value; }
      if (value === null || value === undefined) { return ''; }
      return JSON.stringify(value);
    }

    function setStatus(message) {
      statusEl.textContent = message;
    }

    function addRow(cls, parts) {
      // Event payloads contain model output, so every value goes in through
      // textContent. innerHTML is never used with untrusted content.
      const row = document.createElement('li');
      row.className = cls;
      parts.forEach(function (part) {
        const span = document.createElement('span');
        span.className = 'tag';
        span.textContent = text(part);
        row.appendChild(span);
      });
      listEl.appendChild(row);
      listEl.scrollTop = listEl.scrollHeight;
    }

    function cycleLabel(data) {
      return typeof data.cycle === 'number' ? 'cycle ' + data.cycle : 'cycle ?';
    }

    function showTasks(data) {
      const tasks = Array.isArray(data.tasks) ? data.tasks : [];
      tasks.forEach(function (task) {
        if (!task || typeof task !== 'object') { return; }
        addRow('', [
          cycleLabel(data),
          'task ' + text(task.id) + ' [' + text(task.status || 'pending') + ']',
          text(task.description || '')
        ]);
      });
    }

    function showTests(data) {
      const verdict = data.passed ? 'passed' : 'failed';
      addRow(data.passed ? 'ok' : 'warn', [
        'tests_completed', cycleLabel(data), verdict + ': ' + text(data.summary || '')
      ]);
      const failures = Array.isArray(data.failures) ? data.failures : [];
      failures.forEach(function (failure) {
        addRow('warn', ['failure', cycleLabel(data), text(failure)]);
      });
    }

    function render(data) {
      if (!data || typeof data !== 'object') {
        addRow('bad', ['bad message', text(data)]);
        return;
      }
      const name = typeof data.event === 'string' ? data.event : 'unknown';
      const cycle = cycleLabel(data);
      switch (name) {
        case 'loop_started':
          addRow('info', [name, cycle, text(data.goal)]);
          setStatus('Running: ' + text(data.goal));
          break;
        case 'cycle_started':
          addRow('info', [name, cycle, text(data.focus || 'no focus yet')]);
          break;
        case 'plan_created':
          addRow('info', [name, cycle, text(data.rationale || 'no rationale')]);
          showTasks(data);
          break;
        case 'task_started':
          addRow('info', [name, cycle, 'task ' + text(data.task_id), text(data.description)]);
          break;
        case 'task_completed':
          addRow('ok', [name, cycle, 'task ' + text(data.task_id), text(data.result || '')]);
          break;
        case 'task_failed':
          addRow('bad', [
            name, cycle, 'task ' + text(data.task_id),
            text(data.error || 'failed') + (data.blocked ? ' (blocked)' : '')
          ]);
          break;
        case 'tests_completed':
          showTests(data);
          break;
        case 'cycle_failed':
          addRow('bad', [
            name, cycle, text(data.error || 'cycle failed') + '\n' +
            text(data.traceback || '').slice(0, 2000)
          ]);
          break;
        case 'stop_requested':
        case 'stopping':
          addRow('warn', [name, cycle, 'stop requested']);
          setStatus('Stopping...');
          break;
        case 'loop_stopped':
          addRow('info', [name, cycle, text(data.reason || 'no reason given')]);
          break;
        case 'run_finished':
          addRow('info', [name, cycle, text(data.reason || 'done')]);
          setStatus('Finished: ' + text(data.reason || 'done'));
          break;
        case 'error':
          failed = true;
          addRow('bad', [name, cycle, text(data.error || 'unknown error')]);
          setStatus('Error: ' + text(data.error || 'unknown error'));
          break;
        default:
          addRow('', [name, cycle, text(data)]);
      }
    }

    function setRunning(running) {
      startBtn.disabled = running;
      stopBtn.disabled = !running;
      if (!running) { socket = null; }
    }

    form.addEventListener('submit', function (event) {
      event.preventDefault();
      const goal = goalInput.value.trim();
      if (!goal) {
        failed = true;
        setStatus('Error: enter a goal first.');
        return;
      }
      if (socket) {
        setStatus('A run is already in progress; press Stop first.');
        return;
      }
      failed = false;
      listEl.textContent = '';
      setStatus('Connecting...');
      const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(scheme + '://' + location.host + '/ws');
      socket = ws;
      ws.onopen = function () {
        setRunning(true);
        setStatus('Running...');
        const payload = { goal: goal };
        if (modelInput.value.trim()) { payload.model = modelInput.value.trim(); }
        ws.send(JSON.stringify(payload));
      };
      ws.onmessage = function (event) {
        let data = null;
        try {
          data = JSON.parse(event.data);
        } catch (err) {
          addRow('bad', ['unparsable message', String(event.data).slice(0, 500)]);
          return;
        }
        render(data);
      };
      ws.onerror = function () {
        failed = true;
        addRow('bad', ['socket error', 'could not reach the Furrow server']);
        setStatus('Error: could not reach the Furrow server');
      };
      ws.onclose = function (event) {
        setRunning(false);
        const code = event && event.code ? ' (code ' + event.code + ')' : '';
        addRow('', ['closed', 'Disconnected' + code]);
        setStatus(failed ? 'Stopped after an error.' : 'Disconnected.');
      };
    });

    stopBtn.addEventListener('click', function () {
      if (socket && socket.readyState === WebSocket.OPEN) {
        stopBtn.disabled = true;
        setStatus('Stopping...');
        socket.send(JSON.stringify({ action: 'stop' }));
      }
    });
  })();
</script>
</body>
</html>
"""


# -- server entry point -----------------------------------------------------


def _is_loopback(host: str) -> bool:
    """True when ``host`` only accepts connections from this machine."""
    candidate = host.strip().strip("[]").lower()
    if candidate in {"", "localhost"}:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the UI with uvicorn, on loopback unless told otherwise.

    There is no authentication on ``/ws`` and every goal spends the user's API
    key, so a routable host is allowed but warned about loudly.
    """
    if not _is_loopback(host):
        _console.print(
            f"[bold red]WARNING:[/bold red] exposing Furrow on {host}:{port} with "
            "no authentication. Anyone who can reach that port can spend your API "
            "key and edit your workspace. Prefer an SSH tunnel or an authenticating "
            "reverse proxy."
        )
    _console.print(f"Furrow UI on http://{host}:{port}")
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            ws_ping_interval=20,
            ws_ping_timeout=20,
        )
    )
    try:
        server.run()
    except KeyboardInterrupt:
        _console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        # uvicorn handles Ctrl-C itself when it owns the signal handlers; this
        # is the backstop for a KeyboardInterrupt raised some other way.
        server.should_exit = True
