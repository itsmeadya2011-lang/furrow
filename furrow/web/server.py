"""Main web server for Furrow.

This module sets up the FastAPI application with all routes and the WebSocket
endpoint, and provides the ``run`` function to start the server.
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

POLL_INTERVAL = 0.05

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow</title>
<style>
  :root { 
    color-scheme: dark; 
    --bg-primary: #10131a;
    --bg-secondary: #171b23;
    --bg-tertiary: #1d2430;
    --border-color: #2b3240;
    --text-primary: #d7dae0;
    --text-secondary: #8a93a5;
    --text-muted: #5c6b80;
    --accent-blue: #4c8dff;
    --accent-green: #57d38c;
    --accent-yellow: #f0b367;
    --accent-red: #e0685f;
    --accent-purple: #c084fc;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg-primary: #f8f9fa;
      --bg-secondary: #ffffff;
      --bg-tertiary: #f1f3f5;
      --border-color: #dee2e6;
      --text-primary: #212529;
      --text-secondary: #6c757d;
      --text-muted: #adb5bd;
      --accent-blue: #0d6efd;
      --accent-green: #198754;
      --accent-yellow: #fd7e14;
      --accent-red: #dc3545;
      --accent-purple: #6f42c1;
    }
  }
  body {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    margin: 0; padding: 1.25rem; background: var(--bg-primary); color: var(--text-primary);
    transition: background-color 0.2s, color 0.2s;
  }
  h1 {
    font-size: 1.3rem; letter-spacing: .14em; text-transform: uppercase;
    margin: 0 0 1rem; color: var(--text-primary);
    display: flex; align-items: center; gap: 0.75rem;
  }
  .theme-toggle {
    background: var(--bg-tertiary); border: 1px solid var(--border-color);
    border-radius: 6px; padding: 0.35rem 0.7rem; font-size: 0.85rem;
    color: var(--text-primary); cursor: pointer;
  }
  .theme-toggle:hover { border-color: var(--accent-blue); }
  .bar { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .bar > * { flex: 1 1 20rem; }
  input {
    padding: .55rem .7rem; background: var(--bg-secondary); color: inherit;
    border: 1px solid var(--border-color); border-radius: 6px; font: inherit;
    min-width: 0;
  }
  input:focus { outline: none; border-color: var(--accent-blue); }
  button {
    padding: .55rem 1.1rem; border-radius: 6px; border: 1px solid var(--border-color);
    background: var(--bg-tertiary); color: inherit; font: inherit; cursor: pointer;
    white-space: nowrap;
  }
  button:hover { border-color: var(--accent-blue); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  #log {
    background: var(--bg-primary); border: 1px solid var(--border-color); border-radius: 6px;
    padding: .75rem; height: 20rem; overflow: auto;
    white-space: pre-wrap; word-break: break-word; margin: 0;
    font-size: 0.85rem; line-height: 1.5;
  }
  .header {
    border: 1px solid var(--border-color); border-radius: 8px; padding: .75rem 1rem;
    margin-bottom: 1rem; background: var(--bg-secondary);
  }
  .header .goal {
    font-size: 1.1rem; font-weight: 700; color: var(--text-primary);
    word-break: break-word; margin-bottom: 0.5rem;
  }
  .header .meta {
    display: flex; gap: 1rem; flex-wrap: wrap; margin-top: .5rem;
    font-size: .9rem; color: var(--text-secondary);
  }
  .header .meta b { color: var(--text-primary); }
  .header .reason { color: var(--accent-yellow); }
  .progress-bar {
    height: 6px; background: var(--bg-tertiary); border-radius: 3px;
    overflow: hidden; margin-top: 0.5rem;
  }
  .progress-bar-inner {
    height: 100%; background: var(--accent-blue); border-radius: 3px;
    transition: width 0.3s ease; width: 0%;
  }
  .panels { display: flex; flex-direction: column; gap: .5rem; margin-top: 1rem; }
  .panel {
    border: 1px solid var(--border-color); border-radius: 8px; padding: .6rem .8rem;
    background: var(--bg-secondary);
    transition: border-color 0.2s;
  }
  .panel:hover { border-color: var(--accent-blue); }
  .panel-head {
    display: flex; gap: .5rem; align-items: baseline; margin-bottom: .3rem;
    flex-wrap: wrap;
  }
  .panel-id { font-size: .85rem; color: var(--text-secondary); }
  .panel-status {
    font-size: .75rem; font-weight: 700; text-transform: uppercase;
    padding: .05rem .4rem; border-radius: 3px;
    display: inline-flex; align-items: center; gap: 0.25rem;
  }
  .panel-status::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%;
    background: currentColor;
  }
  .panel-desc {
    margin: 0 0 .4rem; font-size: .95rem; color: var(--text-primary);
    word-break: break-word;
  }
  .panel-meta {
    display: flex; gap: .8rem; font-size: .8rem; color: var(--text-secondary);
    margin-bottom: .4rem; flex-wrap: wrap;
  }
  .panel-files {
    display: flex; flex-wrap: wrap; gap: 0.25rem; margin: 0.25rem 0;
    font-size: 0.75rem;
  }
  .panel-file {
    background: var(--bg-tertiary); padding: 0.1rem 0.4rem;
    border-radius: 3px; color: var(--accent-purple);
    font-family: inherit;
  }
  .panel-result {
    margin: 0 0 .3rem; font-size: .85rem; color: var(--accent-green);
    white-space: pre-wrap; max-height: 100px; overflow: auto;
  }
  .panel-error {
    margin: 0; font-size: .85rem; color: var(--accent-red); white-space: pre-wrap;
    max-height: 100px; overflow: auto;
  }
  .panel-empty {
    color: var(--text-secondary); font-size: .95rem; margin-top: 1rem; text-align: center;
  }
  .status-pending { color: var(--text-secondary); }
  .status-running { color: var(--accent-blue); }
  .status-completed { color: var(--accent-green); }
  .status-failed { color: var(--accent-red); }
  .status-blocked { color: var(--accent-yellow); }
  .status-skipped { color: var(--text-secondary); }
  .status-unknown { color: var(--text-secondary); }
  .log-controls {
    display: flex; gap: 0.5rem; margin-bottom: 0.5rem; flex-wrap: wrap;
    font-size: 0.8rem;
  }
  .log-controls label {
    display: flex; align-items: center; gap: 0.25rem; cursor: pointer;
    color: var(--text-secondary);
  }
  .log-controls input[type="checkbox"] { width: 14px; height: 14px; accent-color: var(--accent-blue); }
  .copy-btn {
    font-size: 0.7rem; padding: 0.15rem 0.4rem;
    background: transparent; border: 1px solid var(--border-color);
  }
  .copy-btn:hover { background: var(--accent-blue); color: white; border-color: var(--accent-blue); }
  .task-dependencies {
    font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;
  }
  .stats-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 0.5rem; margin-top: 0.5rem; padding-top: 0.5rem;
    border-top: 1px solid var(--border-color);
  }
  .stat-item {
    background: var(--bg-tertiary); padding: 0.5rem; border-radius: 6px;
    text-align: center;
  }
  .stat-value { font-size: 1.25rem; font-weight: 700; color: var(--text-primary); }
  .stat-label { font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase; }
  .shortcut-hint {
    font-size: 0.7rem; color: var(--text-muted); margin-left: auto;
    display: flex; gap: 1rem; align-items: center;
  }
  .shortcut-hint kbd {
    background: var(--bg-tertiary); border: 1px solid var(--border-color);
    padding: 0.1rem 0.3rem; border-radius: 3px; font-family: inherit;
  }
</style>
</head>
<body>
<h1>Furrow <span class="shortcut-hint"><kbd>Ctrl</kbd>+<kbd>Enter</kbd> Start | <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Enter</kbd> Resume</span></h1>
<div class="bar">
  <input id="goal" placeholder="What should Furrow build?" autofocus />
  <button id="start" title="Start new run (Ctrl+Enter)">Start</button>
  <button id="resume" title="Resume previous run (Ctrl+Shift+Enter)">Resume</button>
  <button id="theme-toggle" class="theme-toggle" title="Toggle theme">🌙</button>
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
    <div class="progress-bar-inner" id="progress-bar-inner"></div>
  </div>
</div>
<div class="log-controls">
  <label><input type="checkbox" id="log-filter-errors" /> Errors only</label>
  <label><input type="checkbox" id="log-filter-tasks" /> Tasks only</label>
  <label><input type="checkbox" id="log-filter-tests" /> Tests only</label>
  <button id="log-clear" style="margin-left: auto; font-size: 0.75rem;">Clear</button>
</div>
<pre id="log"></pre>
<div class="panels" id="tasks"></div>
<div class="stats-grid" id="stats-grid" style="display: none;">
  <div class="stat-item"><div class="stat-value" id="stat-cycles">0</div><div class="stat-label">Cycles</div></div>
  <div class="stat-item"><div class="stat-value" id="stat-completed">0</div><div class="stat-label">Completed</div></div>
  <div class="stat-item"><div class="stat-value" id="stat-failed">0</div><div class="stat-label">Failed</div></div>
  <div class="stat-item"><div class="stat-value" id="stat-blocked">0</div><div class="stat-label">Blocked</div></div>
  <div class="stat-item"><div class="stat-value" id="stat-cost">$0.00</div><div class="stat-label">Cost</div></div>
  <div class="stat-item"><div class="stat-value" id="stat-duration">0s</div><div class="stat-label">Duration</div></div>
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
  const progressBar = document.getElementById('progress-bar');
  const progressBarInner = document.getElementById('progress-bar-inner');
  const statsGrid = document.getElementById('stats-grid');
  const statCycles = document.getElementById('stat-cycles');
  const statCompleted = document.getElementById('stat-completed');
  const statFailed = document.getElementById('stat-failed');
  const statBlocked = document.getElementById('stat-blocked');
  const statCost = document.getElementById('stat-cost');
  const statDuration = document.getElementById('stat-duration');
  let socket = null;
  let lastSummary = null;
  let runStartTime = null;
  let durationInterval = null;
  const logFilters = {
    errors: document.getElementById('log-filter-errors'),
    tasks: document.getElementById('log-filter-tasks'),
    tests: document.getElementById('log-filter-tests'),
  };

  function append(line, type = 'log') {
    const span = document.createElement('span');
    span.dataset.type = type;
    span.textContent = line + '\n';
    logEl.appendChild(span);
    logEl.scrollTop = logEl.scrollHeight;
    applyLogFilters();
  }

  function applyLogFilters() {
    const showErrors = logFilters.errors.checked;
    const showTasks = logFilters.tasks.checked;
    const showTests = logFilters.tests.checked;
    const anyFilter = showErrors || showTasks || showTests;
    
    logEl.querySelectorAll('span[data-type]').forEach(span => {
      const type = span.dataset.type;
      let visible = true;
      if (anyFilter) {
        visible = false;
        if (showErrors && type === 'error') visible = true;
        if (showTasks && (type === 'task_started' || type === 'task_finished')) visible = true;
        if (showTests && (type === 'test_started' || type === 'test_finished')) visible = true;
      }
      span.style.display = visible ? '' : 'none';
    });
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
    return SAFE_STATUSES[key] || 'unknown';
  }

  function formatDuration(seconds) {
    if (seconds < 60) return seconds + 's';
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins < 60) return mins + 'm ' + secs + 's';
    const hours = Math.floor(mins / 60);
    return hours + 'h ' + (mins % 60) + 'm';
  }

  function updateDuration() {
    if (runStartTime) {
      const elapsed = Math.floor((Date.now() - runStartTime) / 1000);
      statDuration.textContent = formatDuration(elapsed);
    }
  }

  function startDurationTimer() {
    if (durationInterval) clearInterval(durationInterval);
    durationInterval = setInterval(updateDuration, 1000);
    updateDuration();
  }

  function stopDurationTimer() {
    if (durationInterval) clearInterval(durationInterval);
    durationInterval = null;
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
    
    // Update stats grid
    if (status || result) {
      statsGrid.style.display = 'grid';
      statCycles.textContent = text(status && status.cycles) || text(result && result.cycles) || '0';
      statCompleted.textContent = text(result && result.tasks_completed) || '0';
      statFailed.textContent = text(result && result.tasks_failed) || '0';
      statBlocked.textContent = text(result && result.tasks_blocked) || '0';
      statCost.textContent = '$' + (Number(usage.cost_usd) || 0).toFixed(4);
    }
    
    // Progress bar
    if (running && status && status.cycles) {
      progressBar.style.display = 'block';
      // We don't know max cycles, so just show activity
      progressBarInner.style.width = '100%';
      progressBarInner.style.background = 'linear-gradient(90deg, var(--accent-blue), var(--accent-purple))';
    } else {
      progressBar.style.display = 'none';
      progressBarInner.style.width = '0%';
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
      
      // Files changed
      const filesChanged = task.files_changed || [];
      if (filesChanged.length > 0) {
        const filesDiv = document.createElement('div');
        filesDiv.className = 'panel-files';
        filesChanged.forEach(function(file) {
          const fileEl = document.createElement('span');
          fileEl.className = 'panel-file';
          fileEl.textContent = file;
          fileEl.title = 'Click to copy path';
          fileEl.onclick = function() {
            navigator.clipboard.writeText(file);
            const original = fileEl.textContent;
            fileEl.textContent = 'Copied!';
            setTimeout(() => fileEl.textContent = original, 1000);
          };
          filesDiv.appendChild(fileEl);
        });
        panel.appendChild(filesDiv);
      }

      // Dependencies
      const deps = task.dependencies || [];
      if (deps.length > 0) {
        const depsDiv = document.createElement('div');
        depsDiv.className = 'task-dependencies';
        depsDiv.textContent = 'depends on: ' + deps.join(', ');
        panel.appendChild(depsDiv);
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

  async function loadState() {
    try {
      const res = await fetch('/api/state');
      const body = await res.json();
      if (body.state && body.state.tasks) { renderTasks(body.state.tasks); }
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
      if (status && status.tasks) { renderTasks(status.tasks); }
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
      lastSummary = null;
      runStartTime = Date.now();
      startDurationTimer();
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
      let logType = 'log';
      if (type === 'cycle_started') {
        line = 'cycle ' + text(data.cycle) + ' started';
        logType = 'cycle';
      } else if (type === 'plan_created') {
        line = 'plan: ' + truncate(data.rationale, 200) + ' (' + text(data.tasks) + ' tasks)';
        logType = 'plan';
      } else if (type === 'task_started') {
        line = 'task ' + text(data.task) + ' started';
        logType = 'task_started';
      } else if (type === 'task_finished') {
        const ok = data.success === true;
        line = 'task ' + text(data.task) + (ok ? ' completed: ' : ' FAILED: ') + truncate(data.summary, 200);
        color = ok ? '#57d38c' : '#e0685f';
        logType = 'task_finished';
      } else if (type === 'test_started') {
        line = 'running tests...';
        logType = 'test_started';
      } else if (type === 'test_finished') {
        const ok = data.passed === true;
        line = (ok ? 'tests passed: ' : 'tests failed: ') + truncate(data.summary, 200);
        color = ok ? '#57d38c' : '#e0685f';
        logType = 'test_finished';
      } else if (type === 'state_saved') {
        line = 'state saved';
      } else if (type === 'log') {
        line = text(msg.message);
      } else if (type === 'error') {
        line = 'ERROR: ' + text(msg.message);
        color = '#e0685f';
        logType = 'error';
      } else if (type === 'done') {
        line = 'finished: ' + text(msg.message);
        color = '#57d38c';
        logType = 'done';
        stopDurationTimer();
        progressBar.style.display = 'none';
      } else {
        try { line = JSON.stringify(msg); } catch (e) { line = String(msg); }
      }
      const span = document.createElement('span');
      span.style.color = color;
      span.dataset.type = logType;
      span.textContent = time + ' ' + line;
      logEl.appendChild(span);
      logEl.appendChild(document.createTextNode('\n'));
      logEl.scrollTop = logEl.scrollHeight;
      applyLogFilters();
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
      stopDurationTimer();
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
  document.getElementById('log-clear').onclick = function () {
    logEl.textContent = '';
  };
  
  // Log filter handlers
  Object.values(logFilters).forEach(cb => {
    cb.onchange = applyLogFilters;
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
      if (e.shiftKey) {
        document.getElementById('resume').click();
      } else {
        document.getElementById('start').click();
      }
      e.preventDefault();
    }
  });

  // Theme toggle
  const themeToggle = document.getElementById('theme-toggle');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  let isDark = prefersDark;
  themeToggle.textContent = isDark ? '☀️' : '🌙';
  themeToggle.onclick = function() {
    isDark = !isDark;
    document.documentElement.style.colorScheme = isDark ? 'dark' : 'light';
    themeToggle.textContent = isDark ? '☀️' : '🌙';
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


async def _drain(queue: asyncio.Queue[Event]) -> None:
    """Consume events so a background run does not fill its queue forever."""
    while True:
        await queue.get()


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


async def _clear_running_flag(task: asyncio.Task[RunResult]) -> None:
    """Clear the "running" flag when ``task`` is the run that owns the registry."""
    if _CURRENT.get("task") is task:
        _CURRENT["running"] = False


async def _cancel_run(task: asyncio.Task[RunResult]) -> None:
    """Cancel a run and make sure the server does not stay stuck in "running"."""
    task.cancel()
    if _CURRENT.get("task") is task:
        _CURRENT["running"] = False
        _CURRENT["error"] = "run cancelled"


app = FastAPI(title="Furrow")


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
        await _clear_running_flag(run_task)
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
            await _cancel_run(run_task)
        raise
    except (WebSocketDisconnect, RuntimeError):
        if run_task is not None and not run_task.done():
            await _cancel_run(run_task)
    finally:
        bus.unsubscribe(queue)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the Furrow web server."""
    uvicorn.run(app, host=host, port=port)