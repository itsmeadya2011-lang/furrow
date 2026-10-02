"""HTTP + WebSocket front end for Furrow.

Runs are driven in the background: ``Orchestrator.run()`` is an unbounded loop, so
it is never awaited inline on a request/socket coroutine.  Instead each run lives
in a module-level registry and is pumped by an ``asyncio`` task, while progress
events reach interested websockets through a small sync-to-async bridge
(``asyncio.Queue.put_nowait`` from the orchestrator's ``on_event`` callback, a
separate coroutine draining into the socket).

The wire protocol is JSON text frames:

* client -> server (first frame on ``/ws``): ``{"goal": ..., "model"?: ...}`` to
  start a new run, or ``{"run_id": "..."}`` to attach to an existing one.
* server -> client: the orchestrator events verbatim
  (``{"type", "cycle", "message", "data"}``), preceded by one control frame
  (``run_started`` / ``run_attached``) and, on failure, an ``error`` frame.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, settings_for_model
from furrow.core.orchestrator import Orchestrator

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")

#: Starlette reports a gone socket as ``WebSocketDisconnect`` from ``receive``
#: and as ``WebSocketDisconnected`` (a ``RuntimeError``) once it refuses to send
#: again.  Both mean "stop streaming", never "the run failed".
_SOCKET_CLOSED: tuple[type[Exception], ...] = (WebSocketDisconnect, RuntimeError)

#: Per-subscriber back-pressure.  A slow socket loses its *oldest* frames, never
#: the newest ones (so terminal events always arrive).
_QUEUE_SIZE = 512

#: Replay-buffer depth, see ``_RunStream.history``.  One cycle emits roughly
#: ``4 + 2 * len(tasks)`` events, so 256 frames is a dozen-odd cycles of replay --
#: far more than the handful a socket that attaches mid-run needs to reconstruct
#: what it missed -- while keeping a retained run's buffer to a few hundred KB.
_HISTORY_SIZE = 256

#: How many *finished* runs stay queryable.  The registries are process-global and
#: nothing else ever cleared them, so a long-lived server grew without bound: a
#: finished run pins its orchestrator, its whole ``CycleRecord`` history and its
#: stream (plus every subscriber queue) for the lifetime of the process.
#: Retention is counted rather than timed on purpose: a time-to-live would need a
#: clock sweep to reclaim a quiet server's runs, and a count gives a hard upper
#: bound on memory that does not depend on how long the process has been up.  The
#: trade-off is that an old run holds its slot until ``_FINISHED_RETENTION`` newer
#: runs finish, so a long pause between runs keeps the previous one readable.
_FINISHED_RETENTION = 50


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


# --------------------------------------------------------------------- run state

#: ``run_id -> orchestrator``.  A plain dict is enough: every mutation happens on
#: the single event loop with no ``await`` in between, so it cannot interleave.
_RUNS: dict[str, Orchestrator] = {}

#: ``run_id -> asyncio.Task`` strong references, so the event loop never garbage
#: collects a running run before it finishes.
_RUN_TASKS: dict[str, asyncio.Task[None]] = {}

#: ``run_id -> fan-out`` used to bridge the orchestrator's sync ``on_event``
#: callback to any number of websocket subscribers.
_STREAMS: dict[str, _RunStream] = {}

#: Finished ``run_id``s in completion order, oldest first -- the eviction queue
#: behind ``_FINISHED_RETENTION``.  Only runs whose task has completed are ever
#: appended, so pruning this deque can never touch a run that is still going.
#: Deliberately unbounded (``deque(maxlen=...)`` would drop ids from the *left*
#: without evicting them from the registries, leaking exactly what this fixes):
#: ``_prune_finished_runs`` keeps its length bounded instead.
_FINISHED_RUNS: deque[str] = deque()


@dataclass
class _Run:
    run_id: str
    orchestrator: Orchestrator
    task: asyncio.Task[None] | None = None


class _RunStream:
    """Fan-out of one run's events to zero or more websocket subscribers.

    ``history`` is a replay buffer, not a convenience.  ``_start_run`` schedules
    the run with ``asyncio.create_task`` and only *then* does the ``/ws`` handler
    reach ``subscribe()``, and the orchestrator emits ``cycle_start`` before its
    first ``await``.  ``publish()`` can only fan out to the subscribers that
    exist at the instant it runs, so without a buffer every real client loses the
    opening frames of every run -- deterministically, not intermittently.  It is
    a bounded ring of the last ``_HISTORY_SIZE`` events, so a multi-cycle run
    cannot grow it without limit, and it is also what lets a socket attach to an
    already-finished run and read what it missed.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any] | None]] = set()
        self.history: list[dict[str, Any]] = []
        self.finished = False

    # -- called from the orchestrator's own coroutine (sync callback) ---------

    def publish(self, event: dict[str, Any]) -> None:
        self.history.append(event)
        if len(self.history) > _HISTORY_SIZE:
            del self.history[: len(self.history) - _HISTORY_SIZE]
        for queue in list(self._subscribers):
            _offer(queue, event)

    def finish(self) -> None:
        self.finished = True
        for queue in list(self._subscribers):
            _offer(queue, None)  # sentinel: no more events for this run

    # -- websocket side ------------------------------------------------------

    def subscribe(self) -> asyncio.Queue[dict[str, Any] | None]:
        """A queue pre-loaded with the replay buffer, then fed live events.

        Both the copy of ``history`` and the registration below are synchronous
        and neither awaits, exactly like ``publish``.  On a single event loop
        that makes the two impossible to interleave, so no event is lost between
        the replay and the subscription -- and none is delivered twice either,
        because events published *after* this point reach the live queue only.
        """
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=_QUEUE_SIZE)
        for event in list(self.history):
            _offer(queue, event)
        if self.finished:
            _offer(queue, None)  # replay everything, then the closing sentinel
        else:
            self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        self._subscribers.discard(queue)


def _offer(queue: asyncio.Queue[Any], item: Any) -> None:
    """Non-blocking ``put`` that drops the oldest item when the queue is full."""
    if queue.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    queue.put_nowait(item)


def _encode(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def _settings_for(model: str | None) -> Settings | None:
    """Copy of the global settings with every agent model set to ``model``.

    A thin wrapper over the shared :func:`furrow.config.settings_for_model`, so
    the console's model box overrides exactly the fields ``furrow start --model``
    does.  Overriding only the generic ``model`` field here would have been a
    dead override: ``PlannerAgent``, ``WorkerAgent`` and ``TesterAgent`` each
    pass their own ``planner_model`` / ``worker_model`` / ``tester_model`` to the
    LLM client, so none of them would ever read it.

    The module-level ``settings`` singleton is never mutated: concurrent runs may
    legitimately want different models.  ``None`` comes back when there is no
    override, so the run keeps using the singleton unchanged.
    """
    return settings_for_model(model)


async def _drive(run: _Run, stream: _RunStream) -> None:
    """Body of the background task: run the loop, then close the stream."""
    try:
        await run.orchestrator.run()
    except asyncio.CancelledError:
        stream.publish(
            {
                "type": "run_halt",
                "cycle": run.orchestrator.cycles,
                "message": "Run cancelled.",
                "data": {},
            }
        )
        raise
    except Exception as exc:  # never let a run failure vanish silently
        logger.exception("run %s aborted", run.run_id)
        stream.publish(
            {
                "type": "run_halt",
                "cycle": run.orchestrator.cycles,
                "message": f"Run aborted after error: {exc}",
                "data": {"error": str(exc)},
            }
        )
    finally:
        stream.finish()


def _is_running(orchestrator: Orchestrator) -> bool:
    try:
        return bool(orchestrator.status().get("running"))
    except Exception:  # a status that cannot be read is not a reason to evict
        logger.warning("run status unavailable; keeping the run registered")
        return True


def _evict(run_id: str) -> None:
    """Forget one finished run: orchestrator, stream and task reference.

    Dropping the task reference matters as much as the rest: ``_RUN_TASKS`` holds
    the only strong reference to the finished task, and a task keeps its own
    coroutine frame (and therefore the orchestrator it closed over) alive until
    nothing points at it.
    """
    _RUNS.pop(run_id, None)
    _STREAMS.pop(run_id, None)
    _RUN_TASKS.pop(run_id, None)


def _prune_finished_runs() -> None:
    """Evict the oldest finished runs once the retention window overflows.

    Only ids in ``_FINISHED_RUNS`` are considered, and that deque is appended to
    by the run task's own done callback, so a run that is still executing is
    never a candidate.  The ``running`` re-check is belt and braces: a run that
    somehow still reports itself as running is kept and the trim moves on to the
    next-newest entry instead of looping forever.

    An id with no orchestrator is dropped from the deque rather than kept.  That
    case means the run is already gone from ``_RUNS`` -- it was evicted earlier,
    or the registries were cleared underneath the deque -- so there is nothing
    left for this trim to reclaim, and keeping the id would pin a string for the
    life of the process while making ``excess`` impossible to satisfy, which is
    the unbounded growth this whole retention window exists to prevent.
    """
    excess = len(_FINISHED_RUNS) - _FINISHED_RETENTION
    if excess <= 0:
        return
    kept: deque[str] = deque()
    for run_id in _FINISHED_RUNS:
        orchestrator = _RUNS.get(run_id)
        if orchestrator is None:
            continue  # already unregistered: drop the id, it has nothing to evict
        if excess > 0 and not _is_running(orchestrator):
            _evict(run_id)
            excess -= 1
        else:
            kept.append(run_id)
    _FINISHED_RUNS.clear()
    _FINISHED_RUNS.extend(kept)


def _on_run_done(run_id: str) -> Callable[[asyncio.Task[None]], None]:
    """Done callback: release the task reference, then trim finished runs."""

    def _finished(task: asyncio.Task[None]) -> None:
        # The only strong reference to a finished task lives in _RUN_TASKS; drop
        # it so the task (and everything its coroutine frame closed over) can be
        # collected instead of living until the process exits.
        if _RUN_TASKS.get(run_id) is task:
            _RUN_TASKS.pop(run_id, None)
        # ``_drive`` has already called ``stream.finish()`` by the time a task is
        # done, so the run is finished and eligible for the retention window.
        _FINISHED_RUNS.append(run_id)
        _prune_finished_runs()

    return _finished


async def _start_run(goal: str, model: str | None = None) -> _Run:
    """Create an orchestrator and drive it from a background task."""
    run_id = uuid4().hex
    stream = _RunStream()
    run = _Run(
        run_id=run_id,
        orchestrator=Orchestrator(
            goal=goal,
            on_event=stream.publish,
            settings=_settings_for(model),
        ),
    )
    _RUNS[run_id] = run.orchestrator
    _STREAMS[run_id] = stream
    run.task = asyncio.create_task(_drive(run, stream), name=f"furrow-run-{run_id}")
    _RUN_TASKS[run_id] = run.task
    run.task.add_done_callback(_on_run_done(run_id))
    return run


def _stream_for(run_id: str) -> _RunStream | None:
    stream = _STREAMS.get(run_id)
    if stream is None and run_id in _RUNS:
        # Registry entry without a stream cannot happen today; recover anyway so
        # an attached socket still gets a status frame instead of nothing.
        stream = _RunStream()
        stream.finished = True
        _STREAMS[run_id] = stream
    return stream


def _snapshot(run_id: str, orchestrator: Orchestrator) -> dict[str, Any]:
    return {"run_id": run_id, **orchestrator.status()}


# --------------------------------------------------------------------- REST API


@app.post("/api/runs", status_code=202)
async def create_run(request: StartRequest) -> dict[str, Any]:
    goal = request.goal.strip()
    if not goal:
        raise HTTPException(status_code=422, detail="goal must not be empty")
    run = await _start_run(goal, request.model)
    return _snapshot(run.run_id, run.orchestrator)


@app.get("/api/runs")
async def list_runs() -> list[dict[str, Any]]:
    return [_snapshot(run_id, orch) for run_id, orch in list(_RUNS.items())]


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    orchestrator = _RUNS.get(run_id)
    if orchestrator is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    return _snapshot(run_id, orchestrator)


# ------------------------------------------------------------------ websockets


async def _send_error(websocket: WebSocket, message: str) -> None:
    """Send a JSON error frame, then close; never raises."""
    # ``message`` keeps the event envelope shape; ``error`` is a plain alias.
    frame = {"type": "error", "cycle": 0, "message": message, "data": {}, "error": message}
    with contextlib.suppress(*_SOCKET_CLOSED):
        await websocket.send_text(_encode(frame))
    with contextlib.suppress(*_SOCKET_CLOSED):
        await websocket.close(code=1008)


async def _first_message(websocket: WebSocket) -> dict[str, Any] | None:
    """The first client frame; ``None`` once the socket has been dealt with."""
    try:
        payload = await websocket.receive_json()
    except WebSocketDisconnect:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        await _send_error(websocket, 'First frame must be JSON: {"goal": ...} or {"run_id": ...}.')
        return None
    if not isinstance(payload, dict):
        await _send_error(websocket, "First frame must be a JSON object.")
        return None
    return payload


async def _pump(websocket: WebSocket, stream: _RunStream) -> None:
    """Forward queued events until the run ends or the client goes away."""
    queue = stream.subscribe()
    try:
        while True:
            event = await queue.get()
            if event is None:
                return
            try:
                await websocket.send_text(_encode(event))
            except _SOCKET_CLOSED:
                return
    finally:
        stream.unsubscribe(queue)
        with contextlib.suppress(*_SOCKET_CLOSED):
            await websocket.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await _first_message(websocket)
        if payload is None:
            return

        run_id = payload.get("run_id")
        goal = str(payload.get("goal") or "").strip()
        if isinstance(run_id, str) and run_id:
            stream = _stream_for(run_id)
            if stream is None:
                await _send_error(websocket, f"Unknown run_id: {run_id}")
                return
            await websocket.send_text(
                _encode(
                    {
                        "type": "run_attached",
                        "run_id": run_id,
                        "message": f"Attached to run {run_id}",
                        "status": _snapshot(run_id, _RUNS[run_id]),
                    }
                )
            )
        elif goal:
            run = await _start_run(goal, payload.get("model"))
            run_id, stream = run.run_id, _STREAMS[run.run_id]
            await websocket.send_text(
                _encode(
                    {
                        "type": "run_started",
                        "run_id": run_id,
                        "message": f"Started run {run_id}",
                        "status": _snapshot(run_id, run.orchestrator),
                    }
                )
            )
        else:
            await _send_error(websocket, 'Send {"goal": "..."} or {"run_id": "..."} to begin.')
            return

        # Blocks only this handler: the run itself lives in _RUN_TASKS, so a
        # disconnect here never stops it.
        await _pump(websocket, stream)
    except WebSocketDisconnect:
        pass
    except RuntimeError:  # socket already gone (starlette WebSocketDisconnected)
        pass


# ------------------------------------------------------------------------- page

_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow</title>
<style>
  :root {
    --bg: #090d0b;
    --panel: #0f1512;
    --line: #1d2a22;
    --fg: #c3d3c8;
    --dim: #6b7f73;
    --green: #4ade80;
    --cyan: #56c8f5;
    --amber: #fbbf24;
    --red: #f87171;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  header {
    display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
    padding: 13px 18px; border-bottom: 1px solid var(--line);
  }
  h1 {
    margin: 0; font-size: 15px; letter-spacing: .22em; text-transform: uppercase;
    color: var(--green); text-shadow: 0 0 12px rgba(74, 222, 128, .35);
  }
  .tag { color: var(--dim); }
  .pill {
    margin-left: auto; padding: 2px 10px; border-radius: 999px;
    border: 1px solid var(--line); color: var(--dim);
  }
  .pill.connecting { color: var(--amber); border-color: var(--amber); }
  .pill.live { color: var(--green); border-color: var(--green); }
  .pill.error { color: var(--red); border-color: var(--red); }
  main { padding: 18px; display: flex; flex-direction: column; gap: 13px; }
  .bar { display: flex; gap: 8px; flex-wrap: wrap; }
  input {
    flex: 1 1 340px; min-width: 180px; padding: 9px 10px;
    background: var(--panel); color: var(--fg);
    border: 1px solid var(--line); border-radius: 6px; font: inherit;
  }
  input:focus { outline: none; border-color: var(--green); }
  input.model { flex: 0 1 210px; }
  button {
    padding: 9px 15px; border: 0; border-radius: 6px; cursor: pointer;
    background: var(--green); color: #06120b; font: inherit; font-weight: 700;
  }
  button.ghost { background: var(--panel); color: var(--fg); border: 1px solid var(--line); font-weight: 400; }
  button:disabled { opacity: .4; cursor: not-allowed; }
  .meta { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; color: var(--dim); }
  .meta code { color: var(--cyan); }
  code { background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: 2px 6px; }
  #log {
    margin: 0; min-height: 330px; max-height: 64vh; overflow: auto; padding: 12px;
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
    white-space: pre-wrap; overflow-wrap: anywhere;
  }
</style>
</head>
<body>
<header>
  <h1>Furrow</h1>
  <span class="tag">autonomous development loop</span>
  <span id="state" class="pill">idle</span>
</header>
<main>
  <div class="bar">
    <input id="goal" autocomplete="off" placeholder="Goal for Furrow, e.g. add a /health endpoint" />
    <input id="model" class="model" autocomplete="off" placeholder="model override (optional)" />
    <button id="start" type="button">Start</button>
    <button id="attach" class="ghost" type="button">Attach</button>
    <button id="detach" class="ghost" type="button" disabled>Detach</button>
  </div>
  <div class="meta">
    <span>run</span><code id="run-id">-</code>
    <span id="cycles"></span><span id="done"></span>
  </div>
  <pre id="log" aria-live="polite"></pre>
</main>
<script>
(function () {
  var goalInput = document.getElementById('goal');
  var modelInput = document.getElementById('model');
  var startBtn = document.getElementById('start');
  var attachBtn = document.getElementById('attach');
  var detachBtn = document.getElementById('detach');
  var stateEl = document.getElementById('state');
  var runIdEl = document.getElementById('run-id');
  var cyclesEl = document.getElementById('cycles');
  var doneEl = document.getElementById('done');
  var logEl = document.getElementById('log');
  var socket = null;
  var runId = '';

  // All server- and model-derived text reaches the DOM through textContent or
  // element properties; nothing is ever parsed as HTML.
  function log(text) {
    logEl.textContent += String(text) + '\n';
    logEl.scrollTop = logEl.scrollHeight;
  }

  function setState(kind, label) {
    stateEl.className = 'pill' + (kind ? ' ' + kind : '');
    stateEl.textContent = label;
  }

  function setRunId(value) {
    runId = typeof value === 'string' ? value : '';
    runIdEl.textContent = runId || '-';
  }

  function clip(value, max) {
    var text;
    if (typeof value === 'string') text = value;
    else if (value === undefined) return '';
    else { try { text = JSON.stringify(value); } catch (err) { return ''; } }
    return text.length > max ? text.slice(0, max) + '...' : text;
  }

  function list(value) {
    return Array.isArray(value) ? value : [];
  }

  function renderStatus(status) {
    if (!status || typeof status !== 'object') return;
    if (typeof status.run_id === 'string') setRunId(status.run_id);
    if (!goalInput.value && typeof status.goal === 'string') goalInput.value = status.goal;
    cyclesEl.textContent = status.cycles === undefined ? '' : 'cycles: ' + status.cycles;
    doneEl.textContent = status.done === true ? 'done'
      : status.running === true ? 'running' : 'idle';
  }

  function renderEvent(ev) {
    if (!ev || typeof ev !== 'object') return;
    var type = typeof ev.type === 'string' ? ev.type : 'event';
    var data = (ev.data && typeof ev.data === 'object') ? ev.data : {};
    var head = '[' + (type + '                    ').slice(0, 12) + '] c' + (ev.cycle || 0)
      + '  ' + clip(ev.message, 300);
    var tail = [];
    if (type === 'plan') {
      if (data.rationale) tail.push('rationale: ' + clip(data.rationale, 220));
      list(data.tasks).forEach(function (task) {
        task = task || {};
        tail.push('* ' + clip(task.id, 40) + '  ' + clip(task.description, 140));
      });
    } else if (type === 'task_start') {
      tail.push(clip(data.description, 220));
    } else if (type === 'task_end') {
      tail.push('status: ' + clip(data.status, 20) + '  ' + clip(data.result, 600));
    } else if (type === 'test_result') {
      tail.push((data.passed === true ? 'PASS ' : 'FAIL ') + clip(data.summary, 220));
      list(data.failures).forEach(function (failure) { tail.push('x ' + clip(failure, 220)); });
    } else if (type === 'cycle_end') {
      tail.push('completed: ' + clip(data.completed, 12) + '  failed: ' + clip(data.failed, 12));
    } else if (type === 'run_started' || type === 'run_attached') {
      renderStatus(ev.status);
    }
    log(head);
    tail.forEach(function (line) { log('             ' + line); });
  }

  function socketUrl() {
    return (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
  }

  function connect(payload, what) {
    if (socket) { try { socket.close(); } catch (err) {} }
    log('--- ' + what);
    setState('connecting', 'connecting');
    var ws = new WebSocket(socketUrl());
    socket = ws;
    detachBtn.disabled = false;
    ws.onopen = function () {
      setState('live', 'streaming');
      ws.send(JSON.stringify(payload));
    };
    ws.onmessage = function (event) {
      var frame;
      try { frame = JSON.parse(event.data); }
      catch (err) { log('unparsable frame: ' + clip(event.data, 120)); return; }
      if (frame && (frame.type === 'error' || frame.error)) {
        log('!! ' + clip(frame.message || frame.error, 300));
        setState('error', 'error');
        return;
      }
      if (frame && typeof frame.run_id === 'string') setRunId(frame.run_id);
      renderEvent(frame);
      if (frame && (frame.type === 'run_complete' || frame.type === 'run_halt')) {
        setState('', 'finished');
      }
    };
    ws.onerror = function () { setState('error', 'error'); };
    ws.onclose = function () {
      socket = null;
      detachBtn.disabled = true;
      if (!stateEl.className.match(/error|finished/)) setState('', 'closed');
      log('--- socket closed');
    };
  }

  function startRun() {
    var goal = goalInput.value.trim();
    if (!goal) { log('!! a goal is required'); goalInput.focus(); return; }
    var payload = { goal: goal };
    var model = modelInput.value.trim();
    if (model) payload.model = model;
    setRunId('');
    connect(payload, 'start: ' + goal);
  }

  function attachRun() {
    if (!runId) { log('!! no run id to attach to yet'); return; }
    connect({ run_id: runId }, 'attach: ' + runId);
  }

  startBtn.addEventListener('click', startRun);
  attachBtn.addEventListener('click', attachRun);
  detachBtn.addEventListener('click', function () {
    if (socket) { socket.close(); log('--- detached (run keeps going)'); }
  });
  goalInput.addEventListener('keydown', function (event) {
    if (event.key === 'Enter') { event.preventDefault(); startRun(); }
  });

  log('Furrow console ready. Enter a goal and press Start.');
}());
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=_INDEX_HTML)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Uvicorn entrypoint (``furrow web``, or ``python -m furrow.web.server``)."""
    uvicorn.run(app, host=host, port=port)
