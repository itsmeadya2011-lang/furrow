"""Web layer for Furrow.

The server exposes a tiny JSON API plus a websocket that streams every
orchestrator event to the browser:

    GET  /            single page UI (inline HTML/CSS/JS)
    GET  /api/state  persisted run state
    GET  /api/status status of the currently active run
    GET  /api/metrics run metrics and statistics
    POST /api/start  start a background run
    WS   /ws         run a goal and stream its events
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from furrow.config import RunResult, Settings, settings
from furrow.core.events import Event, EventBus
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import StateStore
from furrow.logging import correlation_context, get_logger, new_run_id

logger = logging.getLogger(__name__)
web_logger = get_logger("furrow.web")

app = FastAPI(title="Furrow")

POLL_INTERVAL = 0.05

# Metrics tracking
_METRICS = {
    "runs_started": 0,
    "runs_completed": 0,
    "runs_failed": 0,
    "runs_cancelled": 0,
    "total_cycles": 0,
    "total_tasks_completed": 0,
    "total_tasks_failed": 0,
    "total_tasks_blocked": 0,
    "total_tokens_input": 0,
    "total_tokens_output": 0,
    "total_cost_usd": 0.0,
    "run_durations": [],  # seconds
    "start_time": time.time(),
}

DEFAULT_CURRENT: dict[str, Any] = {
    "running": False,
    "orchestrator": None,
    "result": None,
    "task": None,
    "error": None,
}

# Registry of the run that is currently active (or of the last one that ran).
_CURRENT: dict[str, Any] = dict(DEFAULT_CURRENT)

# Strong references to background tasks so they are never garbage collected
# while they are still pending.
_TASKS: set[asyncio.Task[RunResult]] = set()

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow</title>
<style>
  :root { color-scheme: dark; }
  body {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    margin: 0; padding: 1.25rem; background: #10131a; color: #d7dae0;
  }
  h1 {
    font-size: 1.3rem; letter-spacing: .14em; text-transform: uppercase;
    margin: 0 0 1rem; color: #f0f2f5;
  }
  .bar { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1rem; }
  input {
    flex: 1 1 20rem; padding: .55rem .7rem; background: #171b23; color: inherit;
    border: 1px solid #2b3240; border-radius: 6px; font: inherit;
  }
  button {
    padding: .55rem 1.1rem; border-radius: 6px; border: 1px solid #2b3240;
    background: #1d2430; color: inherit; font: inherit; cursor: pointer;
  }
  button:hover { border-color: #4c8dff; }
  #log {
    background: #0b0e13; border: 1px solid #2b3240; border-radius: 6px;
    padding: .75rem; height: 20rem; overflow: auto;
    white-space: pre-wrap; word-break: break-word; margin: 0;
  }
  .header {
    border: 1px solid #2b3240; border-radius: 8px; padding: .75rem 1rem;
    margin-bottom: 1rem; background: #171b23;
  }
  .header .goal {
    font-size: 1.1rem; font-weight: 700; color: #f0f2f5;
    word-break: break-word;
  }
  .header .meta {
    display: flex; gap: 1rem; flex-wrap: wrap; margin-top: .5rem;
    font-size: .9rem; color: #8a93a5;
  }
  .header .meta b { color: #d7dae0; }
  .header .reason { color: #f0b367; }
.panels { display: flex; flex-direction: column; gap: .5rem; margin-top: 1rem; }
  .panel {
    border: 1px solid #2b3240; border-radius: 8px; padding: .6rem .8rem;
    background: #171b23;
  }
  .panel-head {
    display: flex; gap: .5rem; align-items: baseline; margin-bottom: .3rem;
  }
  .panel-id { font-size: .85rem; color: #8a93a5; }
  .panel-status {
    font-size: .8rem; font-weight: 700; text-transform: uppercase;
    padding: .05rem .4rem; border-radius: 3px;
  }
  .panel-desc {
    margin: 0 0 .4rem; font-size: 1rem; color: #d7dae0;
    word-break: break-word;
  }
  .panel-meta {
    display: flex; gap: .8rem; font-size: .85rem; color: #8a93a5;
    margin-bottom: .4rem; flex-wrap: wrap;
  }
  .panel-result {
    margin: 0 0 .3rem; font-size: .9rem; color: #57d38c;
    white-space: pre-wrap;
  }
  .panel-error {
    margin: 0; font-size: .9rem; color: #e0685f; white-space: pre-wrap;
  }
  .panel-empty {
    color: #8a93a5; font-size: .95rem; margin-top: 1rem; text-align: center;
  }
  .status-pending { color: #8a93a5; }
  .status-running { color: #4c8dff; }
  .status-completed { color: #57d38c; }
  .status-failed { color: #e0685f; }
  .status-blocked { color: #f0b367; }
  .status-skipped { color: #8a93a5; }
  .status-unknown { color: #8a93a5; }

  /* Dependency graph styles */
  .graph-container {
    border: 1px solid #2b3240; border-radius: 8px; padding: 1rem;
    background: #171b23; margin-top: 1rem;
  }
  .graph-title {
    font-size: 1rem; font-weight: 700; color: #f0f2f5; margin-bottom: .75rem;
  }
  #depgraph {
    width: 100%; height: 300px; background: #0b0e13;
    border: 1px solid #2b3240; border-radius: 4px;
  }
  .graph-legend {
    display: flex; gap: 1rem; margin-top: .75rem; flex-wrap: wrap;
    font-size: .8rem; color: #8a93a5;
  }
  .legend-item { display: flex; align-items: center; gap: .3rem; }
  .legend-color {
    width: 12px; height: 12px; border-radius: 3px;
  }
</style>
</head>
<body>
<h1>Furrow</h1>
<div class="bar">
  <input id="goal" placeholder="What should Furrow build?" autofocus />
  <button id="start">Start</button>
  <button id="resume">Resume</button>
</div>
<div class="header" id="header">
  <div class="goal" id="goal-display">\u2014</div>
  <div class="meta" id="header-meta">
    <span>cycle <b id="cycle">\u2014</b></span>
    <span>state <b id="state">idle</b></span>
    <span id="stopped-reason" class="reason"></span>
    <span>usage <b id="usage">0 calls, 0 tokens</b></span>
  </div>
</div>
<pre id="log"></pre>
<div class="panels" id="tasks"></div>
<div class="graph-container" id="graph-container" style="display: none;">
  <div class="graph-title">Task Dependency Graph</div>
  <canvas id="depgraph"></canvas>
  <div class="graph-legend" id="graph-legend"></div>
</div>
<script>
  const logEl = document.getElementById('log');
  const goalEl = document.getElementById('goal');
  const tasksEl = document.getElementById('tasks');
  const goalDisplayEl = document.getElementById('goal-display');
  const cycleEl = document.getElementById('cycle');
  const stateEl = document.getElementById('state');
  const stoppedReasonEl = document.getElementById('stopped-reason');
  const usageEl = document.getElementById('usage');
  let socket = null;
  let lastSummary = null;

  function append(line) {
    logEl.textContent += line + '\n';
    logEl.scrollTop = logEl.scrollHeight;
  }

  function text(value) {
    return value === undefined || value === null ? '' : String(value);
  }

  // The task status is planned by the LLM, so it is untrusted input: it is
  // mapped to a known token before it is used to build a class name.
  const SAFE_STATUSES = {
    pending: 'pending',
    running: 'running',
    completed: 'completed',
    failed: 'failed',
    blocked: 'blocked',
    skipped: 'skipped',
    unknown: 'unknown'
  };

  function statusClass(status) {
    const key = text(status).trim().toLowerCase();
    if (Object.prototype.hasOwnProperty.call(SAFE_STATUSES, key)) {
      return SAFE_STATUSES[key];
    }
    return 'unknown';
  }

  function renderStatus(status, result, running) {
    goalDisplayEl.textContent = text(status && status.goal) || '\u2014';
    cycleEl.textContent =
      text(status && status.cycles !== undefined ? status.cycles : '\u2014');
    stateEl.textContent = running ? 'running' : 'stopped';
    const reason =
      text(status && status.stopped_reason) ||
      text(result && result.stopped_reason) || '';
    stoppedReasonEl.textContent = reason;
    stoppedReasonEl.style.display = reason ? '' : 'none';
    const usage = (status && status.usage) || (result && result.usage) || {};
    const calls = Number(usage.calls) || 0;
    const total = (Number(usage.input_tokens) || 0) +
      (Number(usage.output_tokens) || 0);
    const cost = Number(usage.cost_usd) || 0;
    usageEl.textContent = calls + ' calls, ' + total + ' tokens' +
      (cost > 0 ? ', $' + cost.toFixed(6) : '');
  }

  function renderTasks(tasks) {
    while (tasksEl.firstChild) {
      tasksEl.removeChild(tasksEl.firstChild);
    }
    const list = tasks || [];
    if (list.length === 0) {
      const hint = document.createElement('p');
      hint.className = 'panel-empty';
      hint.textContent = 'No run yet \u2014 enter a goal and click Start.';
      tasksEl.appendChild(hint);
      return;
    }
    list.forEach(function (task) {
      const status = text(task.status) || 'pending';

      const panel = document.createElement('div');
      panel.className = 'panel';
      panel.id = 'task-' + text(task.id);

      const head = document.createElement('div');
      head.className = 'panel-head';
      const idEl = document.createElement('span');
      idEl.className = 'panel-id';
      idEl.textContent = '#' + text(task.id);
      const statusEl = document.createElement('span');
      statusEl.className = 'panel-status status-' + statusClass(status);
      statusEl.textContent = status;
      head.appendChild(idEl);
      head.appendChild(statusEl);

      const desc = document.createElement('div');
      desc.className = 'panel-desc';
      desc.textContent = text(task.description);

      const meta = document.createElement('div');
      meta.className = 'panel-meta';
      const attempts = document.createElement('span');
      attempts.textContent = 'attempts: ' + (Number(task.attempts) || 0);
      meta.appendChild(attempts);

      panel.appendChild(head);
      panel.appendChild(desc);
      panel.appendChild(meta);

      const result = text(task.result);
      const error = text(task.error);
      if (result) {
        const resultEl = document.createElement('div');
        resultEl.className = 'panel-result';
        resultEl.textContent = result;
        panel.appendChild(resultEl);
      }
      if (error) {
        const errorEl = document.createElement('div');
        errorEl.className = 'panel-error';
        errorEl.textContent = error;
        panel.appendChild(errorEl);
      }

      tasksEl.appendChild(panel);
    });
  }

  // Dependency graph rendering
  function renderDepGraph(tasks) {
    const container = document.getElementById('graph-container');
    const canvas = document.getElementById('depgraph');
    const legend = document.getElementById('graph-legend');
    if (!tasks || tasks.length === 0) {
      container.style.display = 'none';
      return;
    }
    container.style.display = 'block';

    // Set canvas size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * devicePixelRatio;
    canvas.height = rect.height * devicePixelRatio;
    const ctx = canvas.getContext('2d');
    ctx.scale(devicePixelRatio, devicePixelRatio);

    // Clear
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Status colors
    const statusColors = {
      pending: '#8a93a5',
      running: '#4c8dff',
      completed: '#57d38c',
      failed: '#e0685f',
      blocked: '#f0b367',
      skipped: '#8a93a5',
      unknown: '#8a93a5',
    };

    // Layout tasks in a grid
    const nodeRadius = 20;
    const padding = 50;
    const cols = Math.ceil(Math.sqrt(tasks.length));
    const rows = Math.ceil(tasks.length / cols);
    const cellWidth = (rect.width - 2 * padding) / cols;
    const cellHeight = (rect.height - 2 * padding) / rows;

    const positions = new Map();
    tasks.forEach((task, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const x = padding + col * cellWidth + cellWidth / 2;
      const y = padding + row * cellHeight + cellHeight / 2;
      positions.set(task.id, { x, y });
    });

    // Draw edges (dependencies)
    ctx.lineWidth = 2;
    tasks.forEach(task => {
      const from = positions.get(task.id);
      if (!from) return;
      (task.dependencies || []).forEach(depId => {
        const to = positions.get(depId);
        if (!to) return;
        ctx.strokeStyle = '#4c8dff';
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();

        // Arrow head
        const angle = Math.atan2(to.y - from.y, to.x - from.x);
        const headLen = 10;
        ctx.beginPath();
        ctx.moveTo(to.x - headLen * Math.cos(angle - Math.PI / 6),
                   to.y - headLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(to.x, to.y);
        ctx.lineTo(to.x - headLen * Math.cos(angle + Math.PI / 6),
                   to.y - headLen * Math.sin(angle + Math.PI / 6));
        ctx.fillStyle = '#4c8dff';
        ctx.fill();
      });
    });

    // Draw nodes
    tasks.forEach(task => {
      const pos = positions.get(task.id);
      if (!pos) return;
      const status = text(task.status) || 'pending';
      const color = statusColors[status] || statusColors.unknown;

      // Circle
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, nodeRadius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#2b3240';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Task ID text
      ctx.fillStyle = '#10131a';
      ctx.font = 'bold 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text(task.id), pos.x, pos.y);
    });

    // Draw legend
    legend.innerHTML = '';
    const uniqueStatuses = [...new Set(tasks.map(t => text(t.status) || 'pending'))];
    uniqueStatuses.forEach(status => {
      const item = document.createElement('div');
      item.className = 'legend-item';
      const colorBox = document.createElement('span');
      colorBox.className = 'legend-color';
      colorBox.style.backgroundColor = statusColors[status] || statusColors.unknown;
      const label = document.createElement('span');
      label.textContent = status;
      item.appendChild(colorBox);
      item.appendChild(label);
      legend.appendChild(item);
    });
  }

  async function loadState() {
    try {
      const res = await fetch('/api/state');
      const body = await res.json();
      if (body.state && body.state.tasks) { 
        renderTasks(body.state.tasks);
        renderDepGraph(body.state.tasks);
      }
    } catch (err) {
      append('! could not load state');
    }
  }

  async function loadStatus() {
    try {
      const res = await fetch('/api/status');
      const body = await res.json();
      const running = body.running === true;
      const status = body.status || null;
      const result = body.result || null;
      renderStatus(status, result, running);
      if (status && status.tasks) { 
        renderTasks(status.tasks);
        renderDepGraph(status.tasks);
      }
      if (body.result) {
        const summary = text(body.result.summary);
        if (summary && summary !== lastSummary) {
          append('-- ' + summary);
          lastSummary = summary;
        }
      }
    } catch (err) {
      append('! could not load status');
    }
  }

  function connect(payload) {
    if (socket) { socket.close(); }
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(scheme + '://' + location.host + '/ws');
    socket.onopen = function () {
      append('> connected');
      // A new run starts: forget the previous summary so this run always
      // contributes its own summary line to the log.
      lastSummary = null;
      socket.send(JSON.stringify(payload));
    };
    function truncate(value, max) {
      const s = text(value);
      return s.length > max ? s.slice(0, max) + '\u2026' : s;
    }

    function eventTime(msg) {
      const ts = msg && msg.timestamp;
      if (!ts) { return '--:--:--'; }
      const m = String(ts).match(/T(\d{2}:\d{2}:\d{2})/);
      return m ? m[1] : '--:--:--';
    }

    function renderEvent(msg) {
      const type = msg && msg.type;
      const data = (msg && msg.data) || {};
      const time = eventTime(msg);
      let line = '';
      let color = '#8a93a5';
      if (type === 'cycle_started') {
        line = 'cycle ' + text(data.cycle) + ' started';
      } else if (type === 'plan_created') {
        line = 'plan: ' + truncate(data.rationale, 200) + ' (' + text(data.tasks) + ' tasks)';
      } else if (type === 'task_started') {
        line = 'task ' + text(data.task) + ' started';
      } else if (type === 'task_finished') {
        const ok = data.success === true;
        line = 'task ' + text(data.task) + (ok ? ' completed: ' : ' FAILED: ') +
          truncate(data.summary, 200);
        color = ok ? '#57d38c' : '#e0685f';
      } else if (type === 'test_started') {
        line = 'running tests...';
      } else if (type === 'test_finished') {
        const ok = data.passed === true;
        line = (ok ? 'tests passed: ' : 'tests failed: ') + truncate(data.summary, 200);
        color = ok ? '#57d38c' : '#e0685f';
      } else if (type === 'state_saved') {
        line = 'state saved';
      } else if (type === 'log') {
        line = text(msg.message);
      } else if (type === 'error') {
        line = 'ERROR: ' + text(msg.message);
        color = '#e0685f';
      } else if (type === 'done') {
        line = 'finished: ' + text(msg.message);
        color = '#57d38c';
      } else {
        try { line = JSON.stringify(msg); } catch (e) { line = String(msg); }
      }
      const span = document.createElement('span');
      span.style.color = color;
      span.textContent = time + ' ' + line;
      logEl.appendChild(span);
      logEl.appendChild(document.createTextNode('\n'));
      logEl.scrollTop = logEl.scrollHeight;
      return line;
    }

    socket.onmessage = function (ev) {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch (err) {
        append(ev.data);
        return;
      }
      renderEvent(msg);
      if (msg.type === 'task_started' || msg.type === 'task_finished') {
        loadStatus();
      }
      if (msg.type === 'done') {
        loadStatus();
      }
    };
    socket.onclose = function () {
      append('-- closed');
      socket = null;
      loadStatus();
    };
    socket.onerror = function () {
      append('!! websocket error');
    };
  }

  document.getElementById('start').onclick = function () {
    connect({ goal: goalEl.value, resume: false });
  };
  document.getElementById('resume').onclick = function () {
    connect({ goal: goalEl.value, resume: true });
  };

  loadState();
  loadStatus();
</script>
</body>
</html>
"""


class StartRequest(BaseModel):
    """Body of the ``POST /api/start`` endpoint."""

    goal: str = ""
    max_cycles: int | None = Field(default=None, ge=0)
    resume: bool = False


class SocketRequest(BaseModel):
    """First (and only) message a websocket client sends."""

    goal: str = ""
    max_cycles: int | None = Field(default=None, ge=0)
    resume: bool = False


def _reset_registry() -> None:
    """Forget the active run (used by the tests)."""
    _CURRENT.clear()
    _CURRENT.update(DEFAULT_CURRENT)


def _state_store() -> StateStore:
    """Store for the persisted run state of the current settings."""
    return StateStore(settings.state_path())


def _run_settings(max_cycles: int | None) -> Settings:
    """Settings for a run, with an optional ``max_cycles`` override.

    The merged settings are *validated*: ``model_copy(update=...)`` skips the
    pydantic validation and would happily build a ``Settings`` object with an
    out-of-range value (a negative ``max_cycles`` for instance).  The module
    level ``settings`` object is never modified: a new one is built instead.
    """
    if max_cycles is None:
        return settings
    return Settings.model_validate({**settings.model_dump(), "max_cycles": max_cycles})


def _describe_validation_error(exc: ValidationError) -> str:
    """Short, human readable reason why a payload was rejected."""
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        if location == "max_cycles":
            return "max_cycles must be greater than or equal to 0"
        if location == "goal":
            return "goal must be a string"
    return "invalid request payload"


def _on_run_done(task: asyncio.Task[RunResult]) -> None:
    """Finalise the registry when a run ends."""
    _TASKS.discard(task)
    if _CURRENT.get("task") is not task:
        # Another run owns the registry: never clobber its data, but keep the
        # "running" flag in sync with the run that is actually in flight.
        current = _CURRENT.get("task")
        _CURRENT["running"] = bool(current is not None and not current.done())
        return
    _CURRENT["running"] = False
    if task.cancelled():
        _METRICS["runs_cancelled"] += 1
        _CURRENT["error"] = "run cancelled"
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Furrow run failed: %s", exc)
        _METRICS["runs_failed"] += 1
        _CURRENT["error"] = str(exc)
        return
    result = task.result()
    _CURRENT["result"] = result
    _METRICS["runs_completed"] += 1
    _METRICS["total_cycles"] += result.cycles
    _METRICS["total_tasks_completed"] += result.tasks_completed
    _METRICS["total_tasks_failed"] += result.tasks_failed
    _METRICS["total_tasks_blocked"] += getattr(result, "tasks_blocked", 0) or 0
    usage = getattr(result, "usage", None)
    if usage is not None:
        _METRICS["total_tokens_input"] += usage.input_tokens
        _METRICS["total_tokens_output"] += usage.output_tokens
        _METRICS["total_cost_usd"] += float(getattr(usage, "cost_usd", 0.0) or 0.0)


def _clear_running_flag(task: asyncio.Task[RunResult]) -> None:
    """Clear the "running" flag when ``task`` is the run that owns the registry."""
    if _CURRENT.get("task") is task:
        _CURRENT["running"] = False


def _cancel_run(task: asyncio.Task[RunResult]) -> None:
    """Cancel a run and make sure the server does not stay stuck in "running"."""
    task.cancel()
    if _CURRENT.get("task") is task:
        _CURRENT["running"] = False
        _CURRENT["error"] = "run cancelled"


async def _drain(queue: asyncio.Queue[Event]) -> None:
    """Consume events so a background run does not fill its queue forever."""
    while True:
        await queue.get()


async def _release_client(client: Any) -> None:
    """Release the HTTP resources of an LLM client, when it supports closing.

    ``LLMClient.aclose()`` may not exist yet and may be a plain or an async
    method, so it is looked up defensively and only awaited when the result
    is awaitable.  A failing close is logged and ignored: closing must never
    turn a successful run into a failed one.
    """
    closer = getattr(client, "aclose", None)
    if closer is None:
        return
    try:
        outcome = closer()
        if inspect.isawaitable(outcome):
            await outcome
    except Exception as exc:
        logger.warning("Could not close the LLM client: %s", exc)


async def _run_orchestrator(orchestrator: Orchestrator) -> RunResult:
    """Run an orchestrator and release its LLM client afterwards.

    The client is closed in a ``finally`` block, so its HTTP connections are
    released when the run finishes, when it fails and when it is cancelled
    (e.g. because the websocket client went away).
    """
    start_time = time.time()
    try:
        return await orchestrator.run()
    finally:
        duration = time.time() - start_time
        _METRICS["run_durations"].append(duration)
        await _release_client(getattr(orchestrator, "client", None))


async def _run_and_drain(orchestrator: Orchestrator, bus: EventBus) -> RunResult:
    """Run an orchestrator in the background and discard the emitted events."""
    queue = bus.subscribe()
    drain = asyncio.create_task(_drain(queue))
    try:
        return await _run_orchestrator(orchestrator)
    finally:
        drain.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drain
        bus.unsubscribe(queue)


def _build_orchestrator(
    goal: str, resume: bool, max_cycles: int | None, bus: EventBus
) -> Orchestrator:
    """Create an orchestrator wired to the event bus and the state store."""
    run_settings = _run_settings(max_cycles)
    return Orchestrator(
        goal=goal,
        settings=run_settings,
        state_store=StateStore(run_settings.state_path()),
        event_bus=bus,
        resume=resume,
    )


async def _send_error(websocket: WebSocket, message: str) -> None:
    """Send an error frame to the client and close the connection."""
    await websocket.send_json({"type": "error", "message": message, "data": {}})
    await websocket.close(code=1008)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """The single page UI."""
    return HTMLResponse(content=INDEX_HTML)


@app.get("/api/state")
async def api_state() -> dict[str, Any]:
    """The persisted run state, if there is one."""
    store = _state_store()
    state = store.load()
    return {
        "state": state.model_dump(mode="json") if state is not None else None,
        "exists": store.exists(),
    }


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    """Status of the run that is currently active."""
    orchestrator = _CURRENT.get("orchestrator")
    result = _CURRENT.get("result")
    return {
        "running": bool(_CURRENT.get("running")),
        "status": orchestrator.status() if orchestrator is not None else None,
        "result": result.model_dump(mode="json") if result is not None else None,
        "error": _CURRENT.get("error"),
    }


@app.get("/api/metrics")
async def api_metrics() -> dict[str, Any]:
    """Server metrics and statistics."""
    uptime = time.time() - _METRICS["start_time"]
    avg_duration = 0.0
    if _METRICS["run_durations"]:
        avg_duration = sum(_METRICS["run_durations"]) / len(_METRICS["run_durations"])
    return {
        "uptime_seconds": uptime,
        "runs_started": _METRICS["runs_started"],
        "runs_completed": _METRICS["runs_completed"],
        "runs_failed": _METRICS["runs_failed"],
        "runs_cancelled": _METRICS["runs_cancelled"],
        "total_cycles": _METRICS["total_cycles"],
        "total_tasks_completed": _METRICS["total_tasks_completed"],
        "total_tasks_failed": _METRICS["total_tasks_failed"],
        "total_tasks_blocked": _METRICS["total_tasks_blocked"],
        "total_tokens_input": _METRICS["total_tokens_input"],
        "total_tokens_output": _METRICS["total_tokens_output"],
        "total_cost_usd": _METRICS["total_cost_usd"],
        "average_run_duration_seconds": avg_duration,
    }


@app.get("/api/debug")
async def api_debug() -> dict[str, Any]:
    """Debug information about the current run."""
    orchestrator = _CURRENT.get("orchestrator")
    result = _CURRENT.get("result")
    debug_info = {
        "running": bool(_CURRENT.get("running")),
        "error": _CURRENT.get("error"),
        "settings": {
            "max_cycles": settings.max_cycles,
            "max_parallel_tasks": settings.max_parallel_tasks,
            "max_worker_iterations": settings.max_worker_iterations,
            "max_task_retries": settings.max_task_retries,
            "provider": settings.provider.value,
            "model": settings.model,
            "planner_model": settings.planner_model,
            "worker_model": settings.worker_model,
            "tester_model": settings.tester_model,
        },
    }
    if orchestrator is not None:
        debug_info["orchestrator"] = {
            "goal": orchestrator.goal,
            "original_goal": orchestrator.original_goal,
            "cycles": orchestrator.cycles,
            "tasks_count": len(orchestrator.tasks),
            "active_task_ids": list(orchestrator._active_ids),
            "no_progress": orchestrator._no_progress,
            "prev_sig": orchestrator._prev_sig,
            "stopped_reason": orchestrator._stopped_reason,
            "usage": orchestrator._usage.model_dump() if orchestrator._usage else None,
            "client_usage": orchestrator.client.total_usage.model_dump() if hasattr(orchestrator.client, 'total_usage') else None,
            "client_agent_usage": {k: v.model_dump() for k, v in orchestrator.client.get_all_agent_usage().items()} if hasattr(orchestrator.client, 'get_all_agent_usage') else None,
        }
    if result is not None:
        debug_info["result"] = result.model_dump(mode="json")
    return debug_info


@app.post("/api/start")
async def api_start(request: StartRequest) -> JSONResponse:
    """Start a background run for the given goal."""
    if _CURRENT.get("running"):
        return JSONResponse(
            status_code=409,
            content={"started": False, "error": "a run is already in progress"},
        )
    goal = request.goal.strip()
    if not goal and not request.resume:
        return JSONResponse(
            status_code=400,
            content={"started": False, "error": "goal is required"},
        )
    if not goal:
        state = _state_store().load()
        goal = state.original_goal if state is not None else ""
        if not goal:
            return JSONResponse(
                status_code=400,
                content={"started": False, "error": "no goal to resume"},
            )
    _METRICS["runs_started"] += 1
    web_logger.info("run.started", goal=goal, resume=request.resume)
    bus = EventBus()
    try:
        orchestrator = _build_orchestrator(goal, request.resume, request.max_cycles, bus)
    except ValidationError as validation_error:
        # Defence in depth: the request model already bounds ``max_cycles``,
        # but an invalid run must never start.
        return JSONResponse(
            status_code=400,
            content={"started": False, "error": _describe_validation_error(validation_error)},
        )
    run_task = asyncio.create_task(_run_and_drain(orchestrator, bus))
    _TASKS.add(run_task)
    run_task.add_done_callback(_on_run_done)
    _CURRENT.update(
        {
            "running": True,
            "orchestrator": orchestrator,
            "result": None,
            "task": run_task,
            "error": None,
        }
    )
    return JSONResponse(status_code=200, content={"started": True, "goal": goal})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Run a goal and stream every event it emits to the client."""
    await websocket.accept()
    bus = EventBus()
    queue = bus.subscribe()
    run_task: asyncio.Task[RunResult] | None = None
    try:
        try:
            payload = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        except (RuntimeError, ValueError):
            await _send_error(websocket, "expected a JSON object with a goal")
            return
        try:
            request = SocketRequest.model_validate(payload)
        except ValidationError as validation_error:
            # A rejected payload (an out-of-range ``max_cycles`` for instance)
            # must be answered with an error frame and must never start a run.
            await _send_error(websocket, _describe_validation_error(validation_error))
            return
        except ValueError:
            await _send_error(websocket, "expected a JSON object with a goal")
            return
        goal = request.goal.strip()
        resume = request.resume
        if not goal and not resume:
            await _send_error(websocket, "goal is required")
            return
        if _CURRENT.get("running"):
            # A run is already in flight: refuse to start a second one instead
            # of overwriting the registry entry of the running one.
            await _send_error(websocket, "a run is already in progress")
            return
        _METRICS["runs_started"] += 1
        web_logger.info("run.started", goal=goal, resume=resume)
        try:
            orchestrator = _build_orchestrator(goal, resume, request.max_cycles, bus)
        except ValidationError as validation_error:
            # Defence in depth: the run settings are validated, so an invalid
            # override can never reach the orchestrator.
            await _send_error(websocket, _describe_validation_error(validation_error))
            return
        run_task = asyncio.create_task(_run_orchestrator(orchestrator))
        _TASKS.add(run_task)
        run_task.add_done_callback(_on_run_done)
        _CURRENT.update(
            {
                "running": True,
                "orchestrator": orchestrator,
                "result": None,
                "task": run_task,
                "error": None,
            }
        )
        while True:
            if run_task.done() and queue.empty():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=POLL_INTERVAL)
            except asyncio.TimeoutError:
                continue
            await websocket.send_json(event.to_dict())
        if run_task is None:  # pragma: no cover - defensive
            return
        # The run is over: clear the flag right away instead of waiting for the
        # done-callback to be processed.
        _clear_running_flag(run_task)
        if run_task.cancelled():
            return
        exc = run_task.exception()
        if exc is not None:
            logger.error("Furrow run failed: %s", exc)
            await _send_error(websocket, f"run failed: {exc}")
            return
        result = run_task.result()
        await websocket.send_json(
            {
                "type": "done",
                "message": result.summary,
                "data": result.model_dump(mode="json"),
            }
        )
    except asyncio.CancelledError:
        # The client went away (or the server is shutting down): stop the run
        # and make sure the server does not stay stuck in the "running" state.
        if run_task is not None and not run_task.done():
            _cancel_run(run_task)
        raise
    except (WebSocketDisconnect, RuntimeError):
        if run_task is not None and not run_task.done():
            _cancel_run(run_task)
    finally:
        bus.unsubscribe(queue)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the Furrow web server."""
    uvicorn.run(app, host=host, port=port)
