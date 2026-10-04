"""FastAPI monitoring UI: streams orchestrator progress over a WebSocket.

Shape of the system
-------------------
* Every run is registered in the module-level :data:`_active_runs` dict as a
  :class:`RunRecord`, so concurrent goals are listable, inspectable and stoppable.
* A run is *always* driven by :func:`asyncio.create_task`. The orchestrator does the
  whole plan -> worker -> tester loop, including LLM calls and a test subprocess that may
  block for ``settings.test_timeout`` seconds; running it inline inside a request handler
  would pin the HTTP/WebSocket connection open for the entire run and let a single client
  stall the whole server.
* The orchestrator's ``on_event`` callback is the single source of truth for progress. It
  runs on the event loop thread inside :meth:`EventEmitter.emit`, so it must never block:
  it appends to the record's bounded history and hands the event to an
  :class:`asyncio.Queue`, and a separate ``_pump`` task does the actual socket writes.
  That queue (rather than ``create_task`` per frame) is what guarantees frames reach the
  browser in emission order.
* Every terminal state is reported. A run that raises produces a final ``error`` frame
  before the socket closes; a cancelled run is flagged on the record; a payload that fails
  validation gets a clear error frame instead of an ``AttributeError``.

Security note: the endpoint is UNAUTHENTICATED and the agent it drives writes files and
spawns subprocesses. :func:`run` therefore defaults to loopback and warns loudly otherwise.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from furrow import __version__
from furrow.config import Settings, get_settings
from furrow.core import Event, EventType, Orchestrator, RunSummary

__all__ = [
    "MAX_CONCURRENT_RUNS",
    "MAX_HISTORY_PER_RUN",
    "RunRecord",
    "StartRequest",
    "app",
    "next_run_id",
    "run",
]

logger = logging.getLogger("furrow.web")

#: Cap on events retained per run. The history backs ``GET /api/runs/{id}`` and doubles as
#: the replay buffer, so it is bounded: a run that iterates for hours would otherwise grow
#: the process heap without limit. The oldest events are dropped first -- a long run stays
#: fully observable through the live stream, only its deep history is truncated.
MAX_HISTORY_PER_RUN = 1000

#: Cap on orchestrators running at the same time. Each one burns provider quota and
#: spawns workers/testers, so a runaway tab cannot queue unbounded work: past this limit a
#: new run is rejected with HTTP 429 (or a WS error frame) instead of being accepted and
#: silently starved of resources. Exactly this many runs may be in flight at once.
MAX_CONCURRENT_RUNS = 4

#: How long a *finished* record is kept before :func:`_reap_runs` drops it. Long enough to
#: read a summary after the fact, short enough that a busy day of runs does not accumulate
#: thousands of retained histories.
RUN_RETENTION_SECONDS = 900.0

#: Grace period given to in-flight orchestrator tasks at shutdown. A cancelled task gets
#: this long to unwind (close its LLM client, reap its subprocess) before we stop waiting,
#: so a reload cannot hang forever on a stuck test command.
SHUTDOWN_GRACE_SECONDS = 10.0

#: Queue sentinel telling :func:`_pump` that no further events can arrive.
_STOP: Final = object()

#: Hosts that keep the UI on this machine. Anything else publishes an unauthenticated
#: agent that writes files and spawns subprocesses to the whole network.
_LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "::1", "localhost"})


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

# A raw string, and not an f-string: the JS and CSS below are full of braces and `\n`
# escapes, which an f-string would force us to double and which would silently turn a
# JavaScript newline into a real one. Kept out of the handler so it is a testable constant.
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Furrow</title>
<style>
  body { font: 14px/1.5 ui-monospace, Menlo, Consolas, monospace; margin: 2rem; }
  h1 { font-size: 1.2rem; }
  form { display: flex; gap: .5rem; margin-bottom: 1rem; }
  input { flex: 1; padding: .4rem; }
  button { padding: .4rem 1rem; }
  button[disabled] { opacity: .5; cursor: progress; }
  #out {
    white-space: pre-wrap; border: 1px solid #ccc; background: #fafafa;
    padding: .75rem; max-height: 70vh; overflow: auto;
  }
</style>
</head>
<body>
<h1>Furrow</h1>
<form id="form">
  <input id="goal" placeholder="Describe the goal" required />
  <button id="start" type="submit">Start</button>
</form>
<pre id="out"></pre>
<script>
  const MAX_LINES = 500;
  const form = document.getElementById('form');
  const goal = document.getElementById('goal');
  const button = document.getElementById('start');
  const out = document.getElementById('out');
  let lines = [];
  let running = false;

  function fmt(ev) {
    const c = 'c' + ev.cycle + ' ';
    switch (ev.type) {
      case 'run_start':
        return c + 'START ' + ev.goal + ' @ ' + ev.workspace;
      case 'cycle_start':
        return c + '--- cycle ' + ev.cycle + ' ---';
      case 'plan':
        return c + 'plan: ' + ev.task_count + ' task(s) - ' + ev.rationale;
      case 'task_start':
        return c + '  > ' + ev.task_id + ': ' + ev.description;
      case 'task_end':
        return c + '  < ' + ev.task_id + ' -> ' + ev.status;
      case 'test':
        return c + 'test ' + (ev.passed ? 'PASSED' : 'FAILED') + ': ' + ev.summary;
      case 'cycle_end':
        return c + 'cycle ' + ev.cycle + ' end: ' + ev.completed + ' ok, ' +
          ev.failed + ' failed, ' + ev.skipped + ' skipped';
      case 'run_end':
        return c + 'STOPPED (' + ev.stopped_because + ') after ' + ev.cycles +
          ' cycle(s): ' + ev.completed + ' ok, ' + ev.failed + ' failed, ' +
          ev.skipped + ' skipped';
      case 'log':
        return c + ev.message;
      case 'error':
        return 'ERROR [' + (ev.stage || '?') + '] ' + ev.message;
      default:
        return c + JSON.stringify(ev);
    }
  }

  function render() {
    out.textContent = lines.join('\n');
    out.scrollTop = out.scrollHeight;
  }

  function push(line) {
    lines.push(line);
    if (lines.length > MAX_LINES) {
      lines.splice(0, lines.length - MAX_LINES);
    }
    render();
  }

  function finish(note) {
    if (!running) { return; }
    running = false;
    button.disabled = false;
    push('-- ' + note + ' --');
  }

  form.onsubmit = (e) => {
    e.preventDefault();
    if (running) { return; }
    const text = goal.value.trim();
    if (!text) { push('enter a goal first'); return; }
    running = true;
    button.disabled = true;
    lines = [];
    push('connecting to ' + location.host + ' ...');
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(scheme + '://' + location.host + '/ws');
    ws.onopen = () => {
      push('socket open, starting run');
      ws.send(JSON.stringify({ goal: text }));
    };
    ws.onmessage = (ev) => {
      let data;
      try {
        data = JSON.parse(ev.data);
      } catch (err) {
        push(ev.data);
        return;
      }
      push(fmt(data));
      if (data.type === 'error') { finish('failed'); }
      if (data.type === 'run_end') {
        finish('finished: ' + data.stopped_because);
      }
    };
    ws.onerror = () => { push('socket error'); };
    ws.onclose = (ev) => { finish('closed (code ' + ev.code + ')'); };
  };
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Run registry
# --------------------------------------------------------------------------- #


@dataclass
class RunRecord:
    """Everything the API knows about one orchestrated run.

    Attributes:
        id: Short unique id, also the path segment in ``/api/runs/{run_id}``.
        goal: The validated goal text handed to the orchestrator.
        started_at: :func:`time.monotonic` reading at creation. Monotonic (not wall clock)
            on purpose: it is only ever used for durations and TTLs, and it cannot jump
            backwards when the system clock is adjusted.
        task: The :func:`asyncio.create_task` driving the run, or ``None`` before launch.
        history: Every event emitted, oldest first, trimmed to
            :data:`MAX_HISTORY_PER_RUN`.
        cancelled: Set when the run was stopped via ``POST /api/runs/{id}/stop``, by a
            client disconnecting from ``/ws``, or at shutdown.
        summary: The :class:`RunSummary` returned by a successful run, else ``None``.
        error: Human-readable terminal error for a run that raised, else ``None``.
        finished_at: :func:`time.monotonic` reading when the run reached a terminal state,
            else ``None``. Drives the retention TTL.
    """

    id: str
    goal: str
    started_at: float
    task: asyncio.Task[None] | None = None
    history: list[Event] = field(default_factory=list)
    cancelled: bool = False
    summary: RunSummary | None = None
    error: str | None = None
    finished_at: float | None = None

    @property
    def done(self) -> bool:
        """True once the background task has finished, whatever the outcome."""
        return self.task is not None and self.task.done()

    def elapsed(self) -> float:
        """Seconds since creation, measured to the terminal state once there is one."""
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return round(end - self.started_at, 3)


#: Every run known to this process, keyed by :attr:`RunRecord.id`. Module-level because the
#: routes are module-level functions, so this is the state they share. Only ever mutated on
#: the event loop thread, so no lock is needed.
_active_runs: dict[str, RunRecord] = {}


class StartRequest(BaseModel):
    """A goal to run, shared by ``POST /api/runs`` and the ``/ws`` handshake.

    One model validates both entry points, which is what turns a blank goal or a
    non-object JSON body into a 422 (or a clear WS error frame) instead of an
    ``AttributeError`` raised by ``data.get(...)`` on a list.
    """

    model_config = ConfigDict(extra="ignore")

    goal: str = Field(
        min_length=1,
        description="What the agent should accomplish. Whitespace is trimmed.",
    )
    model: str | None = Field(
        default=None,
        min_length=1,
        description="Override the model id for every role in this run.",
    )
    workspace: str | None = Field(
        default=None,
        min_length=1,
        description="Override the project root for this run.",
    )

    @field_validator("goal", "model", "workspace", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        """Trim surrounding whitespace before ``min_length`` is checked."""
        return value.strip() if isinstance(value, str) else value


def next_run_id() -> str:
    """Return a short unique run id (12 hex chars of a uuid4)."""
    return uuid4().hex[:12]


def _reap_runs() -> None:
    """Drop finished run records older than :data:`RUN_RETENTION_SECONDS`.

    Only *finished* records are eligible: a record survives its whole run plus
    :data:`RUN_RETENTION_SECONDS` afterwards, so a live run is never reaped out from under
    its own task.
    """
    now = time.monotonic()
    stale = [
        run_id
        for run_id, record in _active_runs.items()
        if record.finished_at is not None
        and now - record.finished_at > RUN_RETENTION_SECONDS
    ]
    for run_id in stale:
        del _active_runs[run_id]


def _active_run_count() -> int:
    """Number of runs that have not reached a terminal state yet."""
    return sum(1 for record in _active_runs.values() if not record.done)


def _ordered_runs() -> list[RunRecord]:
    """All known runs, oldest first."""
    return sorted(_active_runs.values(), key=lambda record: record.started_at)


def _lookup(run_id: str) -> RunRecord:
    """Return the run with ``run_id``, or raise a 404."""
    record = _active_runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run id {run_id!r}")
    return record


def _append_event(record: RunRecord, event: Event) -> None:
    """Append ``event`` to ``record``'s history, dropping the oldest past the cap."""
    record.history.append(event)
    overflow = len(record.history) - MAX_HISTORY_PER_RUN
    if overflow > 0:
        del record.history[:overflow]


def _make_subscriber(
    record: RunRecord, sink: Callable[[Event], None] | None
) -> Callable[[Event], None]:
    """Build the ``on_event`` callback for ``record``.

    Retains every event in the bounded history and, when ``sink`` is given (a WebSocket
    stream), forwards it. Must stay non-blocking: it is called from inside
    :meth:`EventEmitter.emit` on the loop thread, so ``sink`` is a ``Queue.put_nowait``.
    """

    def on_event(event: Event) -> None:
        """Record and forward one event. Never raises."""
        _append_event(record, event)
        if sink is not None:
            sink(event)

    return on_event


def _synthetic_event(record: RunRecord, event_type: EventType, message: str) -> Event:
    """Build a web-layer event for ``record``, inheriting its current cycle number."""
    cycle: Any = record.history[-1].get("cycle", 0) if record.history else 0
    return {
        "type": event_type.value,
        "cycle": cycle if isinstance(cycle, int) else 0,
        "stage": "web",
        "message": message,
    }


def _settings_for(start: StartRequest) -> Settings | None:
    """Run-scoped settings for ``start``, or ``None`` to use the process-wide singleton.

    Only built when the request actually overrides something, so the common path hands the
    orchestrator the untouched singleton rather than a deep copy of it. The copy is deep
    because ``get_settings()`` is process-wide and one run must not reconfigure every other
    component in the interpreter (the same rule the CLI follows).
    """
    if start.model is None and start.workspace is None:
        return None
    settings = get_settings().model_copy(deep=True)
    if start.model is not None:
        settings.model = start.model
        settings.planner_model = start.model
        settings.worker_model = start.model
        settings.tester_model = start.model
    if start.workspace is not None:
        settings.workspace = Path(start.workspace)
    return settings


def _launch(
    record: RunRecord,
    *,
    settings: Settings | None,
    sink: Callable[[Event], None] | None,
) -> asyncio.Task[None]:
    """Start ``record``'s orchestrator as a background task and return that task."""
    task = asyncio.create_task(
        _execute_run(record, settings=settings, sink=sink),
        name=f"furrow-run-{record.id}",
    )
    record.task = task
    return task


def _register_run(start: StartRequest, sink: Callable[[Event], None] | None) -> RunRecord:
    """Create, register and launch a run for ``start``; raises 429 past the cap.

    The cap is checked *before* anything is created, so a refused run leaves no record and
    no dangling coroutine behind.
    """
    _reap_runs()
    if _active_run_count() >= MAX_CONCURRENT_RUNS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"{MAX_CONCURRENT_RUNS} runs are already in flight; "
                "stop one before starting another"
            ),
        )
    record = RunRecord(id=next_run_id(), goal=start.goal, started_at=time.monotonic())
    _active_runs[record.id] = record
    _launch(record, settings=_settings_for(start), sink=sink)
    return record


async def _execute_run(
    record: RunRecord,
    *,
    settings: Settings | None,
    sink: Callable[[Event], None] | None,
) -> None:
    """Own one run from construction to terminal state.

    This coroutine *is* the background task, so it is also where the outcome is recorded:
    success stores the :class:`RunSummary`, an exception is logged and pushed as a final
    ``error`` event, and a cancellation is flagged (then re-raised, because swallowing
    ``CancelledError`` would make a killed task look successful).

    An ordinary exception never propagates: a failed run must not take the server down, and
    the supervising handler reads the outcome off ``record``.

    Note that a failure can therefore show *two* ``error`` frames: the real orchestrator
    emits its own (stage ``"run"``) plus a ``run_end`` before re-raising, and this layer adds
    a terminal frame that names the exception type. The duplicate is deliberate -- the
    second one is the guarantee that a run which raises without emitting anything (a fake, a
    construction failure) still surfaces on the wire instead of closing silently.
    """
    if record.cancelled:
        record.finished_at = time.monotonic()
        return
    on_event = _make_subscriber(record, sink)
    try:
        extra: dict[str, Any] = {}
        if settings is not None:
            extra["settings"] = settings
        orchestrator = Orchestrator(record.goal, on_event=on_event, **extra)
        record.summary = await orchestrator.run()
    except asyncio.CancelledError:
        record.cancelled = True
        _emit_synthetic(record, EventType.LOG, "run cancelled", sink)
        raise
    except Exception as exc:
        logger.exception("furrow run %s failed", record.id)
        record.error = f"{type(exc).__name__}: {exc}"
        _emit_synthetic(record, EventType.ERROR, record.error, sink)
    finally:
        record.finished_at = time.monotonic()


def _emit_synthetic(
    record: RunRecord,
    event_type: EventType,
    message: str,
    sink: Callable[[Event], None] | None,
) -> None:
    """Retain a web-layer event on ``record`` and forward it when there is a stream."""
    event = _synthetic_event(record, event_type, message)
    _append_event(record, event)
    if sink is not None:
        sink(event)


# --------------------------------------------------------------------------- #
# Application + lifespan
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Reap stale records on startup and stop every in-flight run on shutdown.

    Shutdown matters: a cancelled orchestrator is what closes the provider client and reaps
    its test subprocess, so without this a reload would orphan LLM calls and child processes.
    """
    _reap_runs()
    try:
        yield
    finally:
        await _shutdown_runs()


app = FastAPI(title="Furrow", version=__version__, lifespan=_lifespan)


async def _shutdown_runs() -> None:
    """Cancel every in-flight orchestrator task and wait briefly for it to unwind."""
    pending: list[asyncio.Task[None]] = []
    for record in _active_runs.values():
        running_task = record.task
        if running_task is not None and not running_task.done():
            pending.append(running_task)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(pending, timeout=SHUTDOWN_GRACE_SECONDS)
    now = time.monotonic()
    for record in _active_runs.values():
        settled = record.task
        if settled is None or not settled.done():
            continue
        if record.finished_at is None:
            record.finished_at = now
        if settled.cancelled():
            record.cancelled = True


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the self-contained monitoring UI."""
    return HTMLResponse(content=INDEX_HTML)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Liveness probe carrying the number of runs currently in flight."""
    return {"status": "ok", "runs": _active_run_count(), "version": __version__}


@app.post("/api/runs")
async def create_run(start: StartRequest) -> dict[str, Any]:
    """Register a run and start it in the background; 429 past :data:`MAX_CONCURRENT_RUNS`."""
    record = _register_run(start, sink=None)
    return {"id": record.id, "goal": record.goal}


def _list_item(record: RunRecord) -> dict[str, Any]:
    """The summary shape shared by the collection and detail routes (``events`` = count)."""
    return {
        "id": record.id,
        "goal": record.goal,
        "started_at": record.started_at,
        "done": record.done,
        "cancelled": record.cancelled,
        "events": len(record.history),
    }


@app.get("/api/runs")
async def list_runs() -> list[dict[str, Any]]:
    """List known runs, oldest first."""
    return [_list_item(record) for record in _ordered_runs()]


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """One run's summary, full event history and terminal state; 404 if the id is unknown."""
    record = _lookup(run_id)
    detail = _list_item(record)
    detail["elapsed"] = record.elapsed()
    detail["error"] = record.error
    detail["history"] = list(record.history)
    detail["summary"] = record.summary.to_dict() if record.summary is not None else None
    return detail


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str) -> dict[str, Any]:
    """Cancel a run's background task; 404 if the id is unknown.

    Idempotent: stopping a run that already finished still reports the requested state, and
    the record keeps whatever terminal summary it had.
    """
    record = _lookup(run_id)
    task = record.task
    if task is not None and not task.done():
        task.cancel()
    record.cancelled = True
    return {"id": record.id, "cancelled": True}


# --------------------------------------------------------------------------- #
# WebSocket stream
# --------------------------------------------------------------------------- #


class _BadStart(Exception):
    """Raised when a WebSocket payload cannot be turned into a :class:`StartRequest`."""


def _describe_validation_error(exc: ValidationError) -> str:
    """Render a pydantic error as one readable line for a WS error frame."""
    parts = [
        f"{'.'.join(str(item) for item in err.get('loc', ())) or 'payload'}: {err.get('msg')}"
        for err in exc.errors()
    ]
    return "; ".join(parts) or "payload is not a valid start request"


async def _read_start_request(websocket: WebSocket) -> StartRequest:
    """Read the first frame and validate it as a :class:`StartRequest`.

    Deliberately not :meth:`WebSocket.receive_json`: that helper returns ``Any``, so a JSON
    array, string or number would reach ``.get`` and raise ``AttributeError`` with no useful
    message. Decoding here means every malformed payload produces a specific error frame.
    """
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise _BadStart("client disconnected before sending a goal")
    raw = message.get("text")
    if raw is None:
        raw = (message.get("bytes") or b"").decode("utf-8", "replace")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _BadStart(f"payload is not valid JSON: {exc}") from exc
    try:
        return StartRequest.model_validate(payload)
    except ValidationError as exc:
        raise _BadStart(_describe_validation_error(exc)) from exc


async def _send_ws_error(websocket: WebSocket, message: str, *, close_code: int) -> None:
    """Send a terminal ``error`` frame and close, never raising on an already-dead socket."""
    event: Event = {
        "type": EventType.ERROR.value,
        "cycle": 0,
        "stage": "web",
        "message": message,
    }
    with suppress(RuntimeError, WebSocketDisconnect):
        await websocket.send_text(json.dumps(event, default=str))
    with suppress(RuntimeError, WebSocketDisconnect):
        await websocket.close(code=close_code)


async def _pump(websocket: WebSocket, queue: asyncio.Queue[object]) -> None:
    """Forward queued events as one JSON text frame each, in emission order.

    Runs until the :data:`_STOP` sentinel, which the supervising handler queues only after
    the run has finished -- so every event is on the wire before the socket closes. Sending
    from a single task (rather than one task per event) is what preserves ordering.
    """
    while True:
        item = await queue.get()
        if item is _STOP:
            return
        try:
            await websocket.send_text(json.dumps(item, default=str))
        except (WebSocketDisconnect, RuntimeError):
            # The client is gone: stop pumping, but let the run finish on its own.
            return


async def _await_disconnect(websocket: WebSocket) -> None:
    """Block until the client closes the socket."""
    while True:
        try:
            message = await websocket.receive()
        except (WebSocketDisconnect, RuntimeError):
            return
        if message["type"] == "websocket.disconnect":
            return


async def _supervise(websocket: WebSocket, run: asyncio.Task[None]) -> bool:
    """Wait for the run to finish or the client to vanish; True if the client vanished."""
    watcher = asyncio.create_task(
        _await_disconnect(websocket), name=f"furrow-ws-watch-{run.get_name()}"
    )
    try:
        done, _pending = await asyncio.wait(
            {run, watcher}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher
    return run not in done


@app.websocket("/ws")
async def websocket_stream(websocket: WebSocket) -> None:
    """Run one goal and stream its progress as JSON text frames.

    The handshake, not the client's first message, decides the outcome: an unusable payload
    or a full run table gets a clear ``error`` frame and a policy close, and only a valid
    goal reaches :func:`_register_run`. The orchestrator itself runs as a background task
    (see :func:`_launch`), so this handler returns as soon as the run is over -- or as soon
    as the client disconnects, which also cancels the run.
    """
    await websocket.accept()
    try:
        start = await _read_start_request(websocket)
    except _BadStart as exc:
        await _send_ws_error(websocket, str(exc), close_code=1008)
        return

    _reap_runs()
    if _active_run_count() >= MAX_CONCURRENT_RUNS:
        await _send_ws_error(
            websocket,
            f"{MAX_CONCURRENT_RUNS} runs are already in flight; retry once one finishes",
            close_code=1013,
        )
        return

    queue: asyncio.Queue[object] = asyncio.Queue()
    record = RunRecord(id=next_run_id(), goal=start.goal, started_at=time.monotonic())
    _active_runs[record.id] = record
    run = _launch(record, settings=_settings_for(start), sink=queue.put_nowait)
    pump = asyncio.create_task(_pump(websocket, queue), name=f"furrow-ws-{record.id}")

    disconnected = False
    try:
        disconnected = await _supervise(websocket, run)
    finally:
        if disconnected:
            run.cancel()
            pump.cancel()
        queue.put_nowait(_STOP)
        with suppress(asyncio.CancelledError):
            await pump
        with suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(host: str = "127.0.0.1", port: int = 8000, *, reload: bool = False) -> None:
    """Serve the UI with uvicorn.

    Defaults to loopback: the endpoint is UNAUTHENTICATED and drives an agent with file-write
    and subprocess access, so binding 0.0.0.0 hands that to your whole network.

    Args:
        host: Interface to bind. Non-loopback values log a loud warning.
        port: TCP port; must be 1-65535.
        reload: Restart on source changes. Handed to uvicorn as an import string, because
            uvicorn cannot watch a live application object.

    Raises:
        ValueError: If ``port`` is outside 1-65535.
    """
    if not 1 <= port <= 65535:
        raise ValueError(f"port must be between 1 and 65535, got {port!r}.")
    if host not in _LOOPBACK_HOSTS:
        logger.warning(
            "Binding %s publishes furrow on every network interface. The endpoint is "
            "unauthenticated and the agent it drives has file-write and subprocess access. "
            "Prefer 127.0.0.1, or put it behind an authenticating proxy.",
            host,
        )
    # uvicorn re-imports the target on every reload, which needs a string, not this object.
    target: Any = "furrow.web.server:app" if reload else app
    uvicorn.run(
        target,
        host=host,
        port=port,
        reload=reload,
        log_level=get_settings().log_level.lower(),
    )
