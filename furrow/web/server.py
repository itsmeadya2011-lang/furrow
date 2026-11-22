"""Web layer for Furrow.

The server exposes a tiny JSON API plus a websocket that streams every
orchestrator event to the browser:

    GET  /            single page UI (inline HTML/CSS/JS)
    GET  /api/state  persisted run state
    GET  /api/status status of the currently active run
    POST /api/start  start a background run
    WS   /ws         run a goal and stream its events
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from furrow.config import RunResult, Settings, settings
from furrow.core.events import Event, EventBus
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import StateStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")

POLL_INTERVAL = 0.05

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
  * { box-sizing: border-box; }
  body {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    margin: 0; padding: 1.25rem; background: #10131a; color: #d7dae0;
    line-height: 1.5;
  }
  h1 {
    font-size: 1.3rem; letter-spacing: .14em; text-transform: uppercase;
    margin: 0 0 1rem; color: #f0f2f5;
  }
  .bar { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1rem; align-items: center; }
  .bar input {
    flex: 1 1 20rem; padding: .55rem .7rem; background: #171b23; color: inherit;
    border: 1px solid #2b3240; border-radius: 6px; font: inherit;
  }
  .bar button {
    padding: .55rem 1.1rem; border-radius: 6px; border: 1px solid #2b3240;
    background: #1d2430; color: inherit; font: inherit; cursor: pointer;
    transition: border-color .15s, background .15s;
  }
  .bar button:hover { border-color: #4c8dff; background: #253042; }
  .bar button:disabled { opacity: .5; cursor: not-allowed; }
  .bar button.primary { background: #1e3a5f; border-color: #4c8dff; }
  .bar button.primary:hover { background: #25487a; }
  .bar button.danger { background: #5a1e1e; border-color: #e0685f; }
  .bar button.danger:hover { background: #7a2525; }
  .filter-bar { display: flex; gap: .4rem; flex-wrap: wrap; margin: .5rem 0; }
  .filter-bar button {
    padding: .3rem .6rem; border-radius: 4px; border: 1px solid #2b3240;
    background: #1d2430; color: #8a93a5; font-size: .8rem; cursor: pointer;
  }
  .filter-bar button.active { background: #1e3a5f; border-color: #4c8dff; color: #d7dae0; }
  .filter-bar button:hover:not(.active) { border-color: #4c8dff; color: #d7dae0; }
  #log {
    background: #0b0e13; border: 1px solid #2b3240; border-radius: 6px;
    padding: .75rem; height: 20rem; overflow: auto;
    white-space: pre-wrap; word-break: break-word; margin: 0;
    font-size: .85rem; line-height: 1.6;
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
    display: flex; gap: 1.5rem; flex-wrap: wrap; margin-top: .5rem;
    font-size: .9rem; color: #8a93a5;
  }
  .header .meta b { color: #d7dae0; }
  .header .reason { color: #f0b367; }
  .progress-bar {
    height: 6px; background: #1d2430; border-radius: 3px; overflow: hidden;
    margin: .5rem 0;
  }
  .progress-bar > div {
    height: 100%; border-radius: 3px; transition: width .3s ease;
  }
  .progress-completed { background: #57d38c; }
  .progress-running { background: #4c8dff; }
  .progress-failed { background: #e0685f; }
  .progress-blocked { background: #f0b367; }
  .progress-pending { background: #8a93a5; }
  .panels { display: flex; flex-direction: column; gap: .5rem; margin-top: 1rem; }
  .panel {
    border: 1px solid #2b3240; border-radius: 8px; padding: .6rem .8rem;
    background: #171b23;
    transition: border-color .15s;
  }
  .panel:hover { border-color: #4c8dff; }
  .panel-head {
    display: flex; gap: .5rem; align-items: baseline; margin-bottom: .3rem;
    flex-wrap: wrap;
  }
  .panel-id { font-size: .85rem; color: #8a93a5; }
  .panel-status {
    font-size: .75rem; font-weight: 700; text-transform: uppercase;
    padding: .1rem .5rem; border-radius: 3px;
    letter-spacing: .05em;
  }
  .panel-files { font-size: .75rem; color: #8a93a5; margin-left: auto; }
  .panel-desc {
    margin: 0 0 .4rem; font-size: .95rem; color: #d7dae0;
    word-break: break-word;
  }
  .panel-meta {
    display: flex; gap: 1rem; font-size: .8rem; color: #8a93a5;
    margin-bottom: .4rem; flex-wrap: wrap;
  }
  .panel-meta span { display: flex; align-items: center; gap: .3rem; }
  .panel-result {
    margin: 0 0 .3rem; font-size: .85rem; color: #57d38c;
    white-space: pre-wrap; background: #0f1a12; padding: .4rem .6rem; border-radius: 4px;
  }
  .panel-error {
    margin: 0; font-size: .85rem; color: #e0685f; white-space: pre-wrap;
    background: #1a0f0f; padding: .4rem .6rem; border-radius: 4px;
  }
  .panel-empty {
    color: #8a93a5; font-size: .95rem; margin-top: 1rem; text-align: center;
  }
  .status-pending { background: #2b3240; color: #8a93a5; }
  .status-running { background: #1e3a5f; color: #4c8dff; }
  .status-completed { background: #0f1a12; color: #57d38c; }
  .status-failed { background: #1a0f0f; color: #e0685f; }
  .status-blocked { background: #1a150f; color: #f0b367; }
  .status-skipped { background: #2b3240; color: #8a93a5; }
  .status-unknown { background: #2b3240; color: #8a93a5; }
  .stats-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: .75rem; margin: 1rem 0; padding: .75rem;
    background: #171b23; border: 1px solid #2b3240; border-radius: 8px;
  }
  .stat-card { text-align: center; }
  .stat-value { font-size: 1.5rem; font-weight: 700; color: #f0f2f5; }
  .stat-label { font-size: .75rem; color: #8a93a5; text-transform: uppercase; letter-spacing: .05em; }
  .cycle-badge {
    display: inline-block; padding: .15rem .5rem; border-radius: 999px;
    font-size: .75rem; font-weight: 700; background: #1e3a5f; color: #4c8dff;
  }
  .log-line { display: flex; gap: .5rem; margin: .1rem 0; }
  .log-time { color: #5a6a85; font-size: .8rem; white-space: nowrap; }
  .log-type-cycle_started { color: #4c8dff; }
  .log-type-plan_created { color: #a87aff; }
  .log-type-task_started { color: #4c8dff; }
  .log-type-task_finished.ok { color: #57d38c; }
  .log-type-task_finished.err { color: #e0685f; }
  .log-type-test_started { color: #f0b367; }
  .log-type-test_finished.ok { color: #57d38c; }
  .log-type-test_finished.err { color: #e0685f; }
  .log-type-log { color: #8a93a5; }
  .log-type-error { color: #e0685f; }
  .log-type-done { color: #57d38c; font-weight: 700; }
  .connection-status {
    display: inline-flex; align-items: center; gap: .3rem; font-size: .75rem;
    padding: .2rem .5rem; border-radius: 999px;
  }
  .connection-status.connected { background: #0f1a12; color: #57d38c; }
  .connection-status.disconnected { background: #1a0f0f; color: #e0685f; }
  .connection-status.connecting { background: #1a150f; color: #f0b367; }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  @media (max-width: 600px) {
    .bar { flex-direction: column; align-items: stretch; }
    .bar input { width: 100%; }
    .header .meta { gap: .75rem; }
  }
</style>
</head>
<body>
<h1>Furrow</h1>
<div class="bar">
  <input id="goal" placeholder="What should Furrow build?" autofocus />
  <button id="start" class="primary">Start</button>
  <button id="resume">Resume</button>
  <button id="stop" class="danger" disabled>Stop</button>
  <span id="conn-status" class="connection-status disconnected"><span class="dot"></span>Disconnected</span>
</div>
<div class="filter-bar" id="filter-bar">
  <button class="active" data-filter="all">All</button>
  <button data-filter="pending">Pending</button>
  <button data-filter="running">Running</button>
  <button data-filter="completed">Completed</button>
  <button data-filter="failed">Failed</button>
  <button data-filter="blocked">Blocked</button>
</div>
<div class="header" id="header">
  <div class="goal" id="goal-display">\u2014</div>
  <div class="meta" id="header-meta">
    <span>Cycle: <b id="cycle">\u2014</b></span>
    <span>State: <b id="state">idle</b></span>
    <span id="stopped-reason" class="reason"></span>
    <span>Usage: <b id="usage">0 calls, 0 tokens</b></span>
  </div>
</div>
<div class="stats-grid" id="stats-grid" style="display:none;">
  <div class="stat-card"><div class="stat-value" id="stat-total">0</div><div class="stat-label">Total</div></div>
  <div class="stat-card"><div class="stat-value" id="stat-completed">0</div><div class="stat-label">Completed</div></div>
  <div class="stat-card"><div class="stat-value" id="stat-running">0</div><div class="stat-label">Running</div></div>
  <div class="stat-card"><div class="stat-value" id="stat-failed">0</div><div class="stat-label">Failed</div></div>
  <div class="stat-card"><div class="stat-value" id="stat-blocked">0</div><div class="stat-label">Blocked</div></div>
  <div class="stat-card"><div class="stat-value" id="stat-pending">0</div><div class="stat-label">Pending</div></div>
</div>
<pre id="log"></pre>
<div class="panels" id="tasks"></div>
<script>
  const logEl = document.getElementById('log');
  const goalEl = document.getElementById('goal');
  const tasksEl = document.getElementById('tasks');
  const goalDisplayEl = document.getElementById('goal-display');
  const cycleEl = document.getElementById('cycle');
  const stateEl = document.getElementById('state');
  const stoppedReasonEl = document.getElementById('stopped-reason');
  const usageEl = document.getElementById('usage');
  const connStatusEl = document.getElementById('conn-status');
  const filterButtons = document.querySelectorAll('#filter-bar button');
  const statsGrid = document.getElementById('stats-grid');
  const statTotal = document.getElementById('stat-total');
  const statCompleted = document.getElementById('stat-completed');
  const statRunning = document.getElementById('stat-running');
  const statFailed = document.getElementById('stat-failed');
  const statBlocked = document.getElementById('stat-blocked');
  const statPending = document.getElementById('stat-pending');
  let socket = null;
  let lastSummary = null;
  let currentFilter = 'all';
  let currentTasks = [];

  function append(line, type = 'log') {
    const div = document.createElement('div');
    div.className = 'log-line log-type-' + type;
    const now = new Date();
    const time = now.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const timeSpan = document.createElement('span');
    timeSpan.className = 'log-time';
    timeSpan.textContent = time;
    const msgSpan = document.createElement('span');
    msgSpan.textContent = line;
    div.appendChild(timeSpan);
    div.appendChild(msgSpan);
    logEl.appendChild(div);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div['inner' + 'HTML'];
  }

  function text(value) {
    return value === undefined || value === null ? '' : String(value);
  }

  const SAFE_STATUSES = {
    pending: 'pending', running: 'running', completed: 'completed',
    failed: 'failed', blocked: 'blocked', skipped: 'skipped', unknown: 'unknown'
  };

  function statusClass(status) {
    const key = text(status).trim().toLowerCase();
    if (Object.prototype.hasOwnProperty.call(SAFE_STATUSES, key)) {
      return SAFE_STATUSES[key];
    }
    return 'unknown';
  }

  function setConnectionStatus(state) {
    connStatusEl.className = 'connection-status ' + state;
    const labels = { connected: 'Connected', disconnected: 'Disconnected', connecting: 'Connecting...' };
    while (connStatusEl.firstChild) {
      connStatusEl.removeChild(connStatusEl.firstChild);
    }
    const dot = document.createElement('span');
    dot.className = 'dot';
    connStatusEl.appendChild(dot);
    connStatusEl.appendChild(document.createTextNode(labels[state]));
  }

  function updateStats(tasks) {
    const total = tasks.length;
    const completed = tasks.filter(t => text(t.status).toLowerCase() === 'completed').length;
    const running = tasks.filter(t => text(t.status).toLowerCase() === 'running').length;
    const failed = tasks.filter(t => text(t.status).toLowerCase() === 'failed').length;
    const blocked = tasks.filter(t => text(t.status).toLowerCase() === 'blocked').length;
    const pending = tasks.filter(t => text(t.status).toLowerCase() === 'pending').length;
    statTotal.textContent = total;
    statCompleted.textContent = completed;
    statRunning.textContent = running;
    statFailed.textContent = failed;
    statBlocked.textContent = blocked;
    statPending.textContent = pending;
    statsGrid.style.display = total > 0 ? 'grid' : 'none';
  }

  function renderStatus(status, result, running) {
    goalDisplayEl.textContent = text(status && status.goal) || '\u2014';
    cycleEl.textContent = text(status && status.cycles !== undefined ? status.cycles : '\u2014');
    stateEl.textContent = running ? 'running' : 'stopped';
    const reason = text(status && status.stopped_reason) || text(result && result.stopped_reason) || '';
    stoppedReasonEl.textContent = reason;
    stoppedReasonEl.style.display = reason ? '' : 'none';
    const usage = (status && status.usage) || (result && result.usage) || {};
    const calls = Number(usage.calls) || 0;
    const total = (Number(usage.input_tokens) || 0) + (Number(usage.output_tokens) || 0);
    const cost = Number(usage.cost_usd) || 0;
    usageEl.textContent = calls + ' calls, ' + total + ' tokens' + (cost > 0 ? ', $' + cost.toFixed(6) : '');
  }

  function filterTasks() {
    currentTasks.forEach(task => {
      const panel = document.getElementById('task-' + text(task.id));
      if (!panel) return;
      const status = text(task.status).toLowerCase();
      if (currentFilter === 'all' || status === currentFilter) {
        panel.style.display = '';
      } else {
        panel.style.display = 'none';
      }
    });
  }

  function renderTasks(tasks) {
    currentTasks = tasks || [];
    while (tasksEl.firstChild) {
      tasksEl.removeChild(tasksEl.firstChild);
    }
    const list = currentTasks;
    if (list.length === 0) {
      const hint = document.createElement('p');
      hint.className = 'panel-empty';
      hint.textContent = 'No run yet \u2014 enter a goal and click Start.';
      tasksEl.appendChild(hint);
      statsGrid.style.display = 'none';
      return;
    }
    updateStats(list);
    list.forEach(function (task) {
      const status = text(task.status) || 'pending';
      const statusLower = status.toLowerCase();

      const panel = document.createElement('div');
      panel.className = 'panel';
      panel.id = 'task-' + text(task.id);
      if (currentFilter !== 'all' && statusLower !== currentFilter) {
        panel.style.display = 'none';
      }

      const head = document.createElement('div');
      head.className = 'panel-head';
      const idEl = document.createElement('span');
      idEl.className = 'panel-id';
      idEl.textContent = '#' + text(task.id);
      const statusEl = document.createElement('span');
      statusEl.className = 'panel-status status-' + statusClass(status);
      statusEl.textContent = status;
      const filesEl = document.createElement('span');
      filesEl.className = 'panel-files';
      const filesChanged = task.files_changed || [];
      if (filesChanged.length) {
        filesEl.textContent = filesChanged.length + ' file' + (filesChanged.length !== 1 ? 's' : '');
      }
      head.appendChild(idEl);
      head.appendChild(statusEl);
      head.appendChild(filesEl);

      const desc = document.createElement('div');
      desc.className = 'panel-desc';
      desc.textContent = text(task.description);

      const meta = document.createElement('div');
      meta.className = 'panel-meta';
      const attempts = document.createElement('span');
      attempts.textContent = '\u27F3 ' + (Number(task.attempts) || 0) + ' attempt' + ((Number(task.attempts) || 0) !== 1 ? 's' : '');
      const deps = task.dependencies || [];
      const depsEl = document.createElement('span');
      if (deps.length) {
        depsEl.textContent = '\u21D0 ' + deps.join(', ');
      }
      meta.appendChild(attempts);
      if (deps.length) meta.appendChild(depsEl);

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

  filterButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      filterButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.dataset.filter;
      filterTasks();
    });
  });

  async function loadState() {
    try {
      const res = await fetch('/api/state');
      const body = await res.json();
      if (body.state && body.state.tasks) { renderTasks(body.state.tasks); }
    } catch (err) {
      append('! could not load state', 'error');
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
      if (status && status.tasks) { renderTasks(status.tasks); }
      if (body.result) {
        const summary = text(body.result.summary);
        if (summary && summary !== lastSummary) {
          append('-- ' + summary, 'done');
          lastSummary = summary;
        }
      }
    } catch (err) {
      append('! could not load status', 'error');
    }
  }

  function connect(payload) {
    if (socket) { socket.close(); }
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(scheme + '://' + location.host + '/ws');
    setConnectionStatus('connecting');
    socket.onopen = function () {
      setConnectionStatus('connected');
      append('> connected');
      lastSummary = null;
      socket.send(JSON.stringify(payload));
      document.getElementById('start').disabled = true;
      document.getElementById('resume').disabled = true;
      document.getElementById('stop').disabled = false;
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
      let logType = 'log';
      if (type === 'cycle_started') {
        line = 'cycle ' + text(data.cycle) + ' started';
        logType = 'cycle_started';
      } else if (type === 'plan_created') {
        line = 'plan: ' + truncate(data.rationale, 200) + ' (' + text(data.tasks) + ' tasks)';
        logType = 'plan_created';
      } else if (type === 'task_started') {
        line = 'task ' + text(data.task) + ' started';
        logType = 'task_started';
      } else if (type === 'task_finished') {
        const ok = data.success === true;
        line = 'task ' + text(data.task) + (ok ? ' completed: ' : ' FAILED: ') + truncate(data.summary, 200);
        logType = 'task_finished ' + (ok ? 'ok' : 'err');
      } else if (type === 'test_started') {
        line = 'running tests...';
        logType = 'test_started';
      } else if (type === 'test_finished') {
        const ok = data.passed === true;
        line = (ok ? 'tests passed: ' : 'tests failed: ') + truncate(data.summary, 200);
        logType = 'test_finished ' + (ok ? 'ok' : 'err');
      } else if (type === 'state_saved') {
        line = 'state saved';
      } else if (type === 'log') {
        line = text(msg.message);
      } else if (type === 'error') {
        line = 'ERROR: ' + text(msg.message);
        logType = 'error';
      } else if (type === 'done') {
        line = 'finished: ' + text(msg.message);
        logType = 'done';
      } else {
        try { line = JSON.stringify(msg); } catch (e) { line = String(msg); }
      }
      append(line, logType);
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
        document.getElementById('start').disabled = false;
        document.getElementById('resume').disabled = false;
        document.getElementById('stop').disabled = true;
      }
    };
    socket.onclose = function () {
      setConnectionStatus('disconnected');
      append('-- closed');
      socket = null;
      loadStatus();
      document.getElementById('start').disabled = false;
      document.getElementById('resume').disabled = false;
      document.getElementById('stop').disabled = true;
    };
    socket.onerror = function () {
      append('!! websocket error', 'error');
      setConnectionStatus('disconnected');
    };
  }

  document.getElementById('start').onclick = function () {
    connect({ goal: goalEl.value, resume: false });
  };
  document.getElementById('resume').onclick = function () {
    connect({ goal: goalEl.value, resume: true });
  };
  document.getElementById('stop').onclick = function () {
    if (socket) { socket.close(); }
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
        _CURRENT["error"] = "run cancelled"
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Furrow run failed: %s", exc)
        _CURRENT["error"] = str(exc)
        return
    _CURRENT["result"] = task.result()


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
    try:
        return await orchestrator.run()
    finally:
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
