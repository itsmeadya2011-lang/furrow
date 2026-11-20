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
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .tabs { display: flex; gap: .25rem; margin-bottom: 1rem; border-bottom: 1px solid #2b3240; }
  .tab {
    padding: .5rem 1rem; background: transparent; border: none; border-bottom: 2px solid transparent;
    color: #8a93a5; font: inherit; cursor: pointer;
  }
  .tab:hover { color: #d7dae0; }
  .tab.active { color: #4c8dff; border-bottom-color: #4c8dff; }
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
  .progress-bar {
    width: 100%; height: 8px; background: #1d2430; border-radius: 4px;
    overflow: hidden; margin: .5rem 0;
  }
  .progress-fill {
    height: 100%; background: linear-gradient(90deg, #4c8dff, #57d38c);
    transition: width 0.3s ease; border-radius: 4px;
  }
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
  
  /* Timeline styles */
  .timeline { display: flex; flex-direction: column; gap: .5rem; }
  .timeline-item {
    display: flex; gap: .75rem; padding: .5rem; background: #171b23;
    border: 1px solid #2b3240; border-radius: 6px;
    position: relative;
  }
  .timeline-item::before {
    content: ''; position: absolute; left: 10px; top: 0; bottom: 0;
    width: 2px; background: #2b3240;
  }
  .timeline-item:first-child::before { top: 50%; }
  .timeline-item:last-child::before { bottom: 50%; }
  .timeline-marker {
    width: 20px; height: 20px; border-radius: 50%; flex-shrink: 0;
    border: 3px solid #10131a; z-index: 1; margin-top: 2px;
  }
  .timeline-content { flex: 1; min-width: 0; }
  .timeline-header { display: flex; gap: .5rem; align-items: center; margin-bottom: .25rem; }
  .timeline-time { font-size: .75rem; color: #8a93a5; white-space: nowrap; }
  .timeline-task { font-weight: 600; color: #f0f2f5; }
  .timeline-details { font-size: .8rem; color: #8a93a5; }
  .timeline-duration { font-family: monospace; }
  
  /* Priority indicators */
  .priority-high { border-left: 3px solid #e0685f; }
  .priority-medium { border-left: 3px solid #f0b367; }
  .priority-low { border-left: 3px solid #4c8dff; }
  
  /* Keyboard shortcut hints */
  .shortcut { font-size: .7rem; padding: .1rem .3rem; background: #2b3240; border-radius: 3px; color: #8a93a5; }
  
  /* Responsive */
  @media (max-width: 600px) {
    .bar { flex-direction: column; }
    .header .meta { flex-direction: column; gap: .25rem; }
  }
</style>
</head>
<body>
<h1>Furrow</h1>
<div class="bar">
  <input id="goal" placeholder="What should Furrow build?" autofocus />
  <button id="start" title="Start (Ctrl+Enter)">Start <span class="shortcut">⌘↵</span></button>
  <button id="resume" title="Resume">Resume</button>
  <button id="stop" title="Stop (Ctrl+Shift+S)" disabled>Stop <span class="shortcut">⌘⇧S</span></button>
</div>
<div class="tabs">
  <button class="tab active" data-tab="tasks">Tasks</button>
  <button class="tab" data-tab="timeline">Timeline</button>
  <button class="tab" data-tab="log">Log</button>
</div>
<div class="header" id="header">
  <div class="goal" id="goal-display">\u2014</div>
  <div class="meta" id="header-meta">
    <span>cycle <b id="cycle">\u2014</b></span>
    <span>state <b id="state">idle</b></span>
    <span id="stopped-reason" class="reason"></span>
    <span>usage <b id="usage">0 calls, 0 tokens</b></span>
  </div>
  <div class="progress-bar" id="progress-bar" style="display: none;">
    <div class="progress-fill" id="progress-fill" style="width: 0%;"></div>
  </div>
</div>
<pre id="log" style="display: none;"></pre>
<div class="panels" id="tasks"></div>
<div class="timeline" id="timeline" style="display: none;"></div>
<script>
  const logEl = document.getElementById('log');
  const goalEl = document.getElementById('goal');
  const tasksEl = document.getElementById('tasks');
  const timelineEl = document.getElementById('timeline');
  const goalDisplayEl = document.getElementById('goal-display');
  const cycleEl = document.getElementById('cycle');
  const stateEl = document.getElementById('state');
  const stoppedReasonEl = document.getElementById('stopped-reason');
  const usageEl = document.getElementById('usage');
  const progressBarEl = document.getElementById('progress-bar');
  const progressFillEl = document.getElementById('progress-fill');
  const startBtn = document.getElementById('start');
  const resumeBtn = document.getElementById('resume');
  const stopBtn = document.getElementById('stop');
  let socket = null;
  let lastSummary = null;
  let currentView = 'tasks';
  let taskTimings = new Map(); // taskId -> {start, end, duration}
  let cycleStartTime = null;

  function append(line) {
    logEl.textContent += line + '\n';
    logEl.scrollTop = logEl.scrollHeight;
  }

  function text(value) {
    return value === undefined || value === null ? '' : String(value);
  }

  function showView(view) {
    currentView = view;
    document.querySelectorAll('.tab').forEach(tab => {
      tab.classList.toggle('active', tab.dataset.tab === view);
    });
    logEl.style.display = view === 'log' ? 'block' : 'none';
    tasksEl.style.display = view === 'tasks' ? 'flex' : 'none';
    timelineEl.style.display = view === 'timeline' ? 'flex' : 'none';
  }

  document.querySelectorAll('.tab').forEach(tab => {
    tab.onclick = () => showView(tab.dataset.tab);
  });

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

  function priorityClass(priority) {
    const p = Number(priority) || 0;
    if (p >= 7) return 'priority-high';
    if (p >= 4) return 'priority-medium';
    return 'priority-low';
  }

  function formatDuration(ms) {
    if (ms < 1000) return ms + 'ms';
    if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
    return (ms / 60000).toFixed(1) + 'm';
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
    
    // Update progress bar
    if (running && status && status.tasks) {
      const total = status.tasks.length;
      const completed = status.tasks.filter(t => t.status === 'completed').length;
      const running_count = status.tasks.filter(t => t.status === 'running').length;
      const progress = total > 0 ? ((completed + running_count * 0.5) / total) * 100 : 0;
      progressBarEl.style.display = 'block';
      progressFillEl.style.width = progress + '%';
    } else if (!running && result) {
      progressBarEl.style.display = 'none';
    }
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
      panel.className = 'panel ' + priorityClass(task.priority);
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
      
      if (task.priority > 0) {
        const priority = document.createElement('span');
        priority.textContent = 'priority: ' + task.priority;
        meta.appendChild(priority);
      }
      if (task.timeout) {
        const timeout = document.createElement('span');
        timeout.textContent = 'timeout: ' + task.timeout + 's';
        meta.appendChild(timeout);
      }
      if (task.dependencies && task.dependencies.length > 0) {
        const deps = document.createElement('span');
        deps.textContent = 'deps: ' + task.dependencies.join(', ');
        meta.appendChild(deps);
      }
      if (task.files_changed && task.files_changed.length > 0) {
        const files = document.createElement('span');
        files.textContent = 'files: ' + task.files_changed.join(', ');
        meta.appendChild(files);
      }

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

  function renderTimeline(tasks) {
    while (timelineEl.firstChild) {
      timelineEl.removeChild(timelineEl.firstChild);
    }
    const list = tasks || [];
    if (list.length === 0) {
      const hint = document.createElement('p');
      hint.className = 'panel-empty';
      hint.textContent = 'No tasks to display in timeline.';
      timelineEl.appendChild(hint);
      return;
    }
    
    // Sort by start time
    const sorted = [...list].sort((a, b) => {
      const aTime = a.started_at ? new Date(a.started_at).getTime() : Infinity;
      const bTime = b.started_at ? new Date(b.started_at).getTime() : Infinity;
      return aTime - bTime;
    });
    
    sorted.forEach(function (task, index) {
      const status = text(task.status) || 'pending';
      const started = task.started_at ? new Date(task.started_at) : null;
      const finished = task.finished_at ? new Date(task.finished_at) : null;
      const duration = started && finished ? finished - started : null;
      
      const item = document.createElement('div');
      item.className = 'timeline-item';
      
      const marker = document.createElement('div');
      marker.className = 'timeline-marker status-' + statusClass(status);
      marker.style.background = statusColor(status);
      item.appendChild(marker);
      
      const content = document.createElement('div');
      content.className = 'timeline-content';
      
      const header = document.createElement('div');
      header.className = 'timeline-header';
      
      const timeEl = document.createElement('span');
      timeEl.className = 'timeline-time';
      timeEl.textContent = started ? started.toLocaleTimeString() : '--:--:--';
      header.appendChild(timeEl);
      
      const taskEl = document.createElement('span');
      taskEl.className = 'timeline-task';
      taskEl.textContent = '#' + text(task.id) + ' ' + text(task.description);
      header.appendChild(taskEl);
      
      content.appendChild(header);
      
      const details = document.createElement('div');
      details.className = 'timeline-details';
      const parts = [];
      parts.push('Status: ' + status);
      if (task.priority > 0) parts.push('Priority: ' + task.priority);
      if (duration !== null) parts.push('Duration: ' + formatDuration(duration));
      if (task.attempts > 1) parts.push('Attempts: ' + task.attempts);
      if (task.files_changed && task.files_changed.length > 0) {
        parts.push('Files: ' + task.files_changed.join(', '));
      }
      details.textContent = parts.join(' | ');
      content.appendChild(details);
      
      item.appendChild(content);
      timelineEl.appendChild(item);
    });
  }

  function statusColor(status) {
    const colors = {
      pending: '#8a93a5',
      running: '#4c8dff',
      completed: '#57d38c',
      failed: '#e0685f',
      blocked: '#f0b367',
      skipped: '#8a93a5',
      unknown: '#8a93a5'
    };
    return colors[status.toLowerCase()] || '#8a93a5';
  }

  async function loadState() {
    try {
      const res = await fetch('/api/state');
      const body = await res.json();
      if (body.state && body.state.tasks) { 
        renderTasks(body.state.tasks);
        renderTimeline(body.state.tasks);
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
        renderTimeline(status.tasks);
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
      startBtn.disabled = true;
      resumeBtn.disabled = true;
      stopBtn.disabled = false;
      // A new run starts: forget the previous summary so this run always
      // contributes its own summary line to the log.
      lastSummary = null;
      taskTimings.clear();
      cycleStartTime = Date.now();
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
        cycleStartTime = Date.now();
      } else if (type === 'plan_created') {
        line = 'plan: ' + truncate(data.rationale, 200) + ' (' + text(data.tasks) + ' tasks)';
      } else if (type === 'task_started') {
        line = 'task ' + text(data.task) + ' started';
        taskTimings.set(data.task, {start: Date.now()});
      } else if (type === 'task_finished') {
        const ok = data.success === true;
        line = 'task ' + text(data.task) + (ok ? ' completed: ' : ' FAILED: ') +
          truncate(data.summary, 200);
        color = ok ? '#57d38c' : '#e0685f';
        const timing = taskTimings.get(data.task);
        if (timing) {
          timing.end = Date.now();
          timing.duration = timing.end - timing.start;
        }
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
        startBtn.disabled = false;
        resumeBtn.disabled = false;
        stopBtn.disabled = true;
      }
    };
    socket.onclose = function () {
      append('-- closed');
      socket = null;
      startBtn.disabled = false;
      resumeBtn.disabled = false;
      stopBtn.disabled = true;
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
  document.getElementById('stop').onclick = function () {
    if (socket) {
      socket.close();
      append('> stop requested');
    }
  };

  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to start
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!startBtn.disabled) startBtn.click();
    }
    // Ctrl/Cmd + Shift + S to stop
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'S') {
      e.preventDefault();
      if (!stopBtn.disabled) stopBtn.click();
    }
    // Tab switching with numbers
    if (e.altKey && ['1', '2', '3'].includes(e.key)) {
      e.preventDefault();
      const tabs = ['tasks', 'timeline', 'log'];
      showView(tabs[parseInt(e.key) - 1]);
    }
  });

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
