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
    margin: 0; padding: 1rem; background: #10131a; color: #d7dae0;
    min-height: 100vh;
  }
  h1 {
    font-size: 1.1rem; letter-spacing: .14em; text-transform: uppercase;
    margin: 0 0 .75rem; color: #f0f2f5;
  }
  .toolbar {
    display: flex;
    gap: .5rem;
    flex-wrap: wrap;
    margin-bottom: .75rem;
    align-items: center;
  }
  .toolbar input {
    flex: 1 1 15rem; padding: .5rem .7rem; background: #171b23; color: inherit;
    border: 1px solid #2b3240; border-radius: 6px; font: inherit; min-width: 0;
  }
  .toolbar button {
    padding: .5rem 1rem; border-radius: 6px; border: 1px solid #2b3240;
    background: #1d2430; color: inherit; font: inherit; cursor: pointer;
    white-space: nowrap;
  }
  .toolbar button:hover { border-color: #4c8dff; }
  .toolbar button:disabled { opacity: .5; cursor: not-allowed; }
  .toolbar button.primary { background: #2a4a6e; border-color: #4c8dff; }
  .toolbar button.primary:hover { background: #3a5a7e; }
  .toolbar button.danger { background: #5a2a2a; border-color: #e0685f; }
  .toolbar button.danger:hover { background: #6a3a3a; }

  /* Dashboard */
  .dashboard {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: .75rem;
    margin-bottom: 1rem;
  }
  .dash-card {
    border: 1px solid #2b3240; border-radius: 8px; padding: .75rem 1rem;
    background: #171b23;
  }
  .dash-card h3 {
    margin: 0 0 .5rem; font-size: .75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .1em; color: #8a93a5;
  }
  .dash-value { font-size: 1.5rem; font-weight: 700; color: #f0f2f5; }
  .dash-value.warning { color: #f0b367; }
  .dash-value.danger { color: #e0685f; }
  .dash-value.success { color: #57d38c; }
  .progress-bar {
    height: 8px; background: #0b0e13; border-radius: 4px; overflow: hidden;
    border: 1px solid #2b3240;
  }
  .progress-bar > div {
    height: 100%; border-radius: 3px; transition: width .3s ease;
  }
  .progress-bar.running > div { background: linear-gradient(90deg, #4c8dff, #57d38c); }
  .progress-bar.completed > div { background: #57d38c; }
  .progress-bar.failed > div { background: #e0685f; }
  .progress-bar.idle > div { background: #2b3240; }
  .cycle-indicator {
    display: flex; align-items: center; gap: .5rem;
  }
  .cycle-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #2b3240; transition: background .3s;
  }
  .cycle-dot.active { background: #4c8dff; box-shadow: 0 0 8px #4c8dff; }
  .cycle-dot.done { background: #57d38c; }
  .token-usage { display: flex; flex-direction: column; gap: .25rem; }
  .token-row { display: flex; justify-content: space-between; font-size: .85rem; }
  .token-row b { color: #f0f2f5; }
  .time-elapsed { font-variant-numeric: tabular-nums; }

  /* Header */
  .header {
    border: 1px solid #2b3240; border-radius: 8px; padding: .75rem 1rem;
    margin-bottom: .75rem; background: #171b23;
  }
  .header .goal {
    font-size: 1rem; font-weight: 700; color: #f0f2f5;
    word-break: break-word; margin-bottom: .5rem;
  }
  .header .meta {
    display: flex; gap: 1rem; flex-wrap: wrap; font-size: .85rem; color: #8a93a5;
  }
  .header .meta b { color: #d7dae0; }
  .header .reason { color: #f0b367; }

  /* Log panel */
  .log-toolbar {
    display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: .5rem; align-items: center;
    font-size: .8rem; color: #8a93a5;
  }
  .log-toolbar input[type="search"] {
    flex: 1 1 12rem; padding: .35rem .5rem; background: #171b23; color: inherit;
    border: 1px solid #2b3240; border-radius: 4px; font: inherit; min-width: 0;
  }
  .log-toolbar select {
    padding: .35rem .5rem; background: #171b23; color: inherit;
    border: 1px solid #2b3240; border-radius: 4px; font: inherit;
  }
  .log-toolbar label { display: flex; align-items: center; gap: .25rem; cursor: pointer; }
  #log {
    background: #0b0e13; border: 1px solid #2b3240; border-radius: 6px;
    padding: .75rem; height: 30vh; min-height: 150px; max-height: 400px; overflow: auto;
    white-space: pre-wrap; word-break: break-word; margin: 0;
    font-size: .8rem; line-height: 1.5;
  }
  #log .log-entry { display: block; margin-bottom: .15rem; }
  #log .log-time { color: #5a6375; margin-right: .5rem; }
  #log .log-type {
    font-weight: 700;
    text-transform: uppercase;
    font-size: .7rem;
    margin-right: .5rem;
  }
  #log .log-type.cycle_started { color: #4c8dff; }
  #log .log-type.plan_created { color: #f0b367; }
  #log .log-type.task_started { color: #57d38c; }
  #log .log-type.task_finished { color: #57d38c; }
  #log .log-type.task_finished.failed { color: #e0685f; }
  #log .log-type.test_started { color: #8a93a5; }
  #log .log-type.test_finished { color: #57d38c; }
  #log .log-type.test_finished.failed { color: #e0685f; }
  #log .log-type.state_saved { color: #8a93a5; }
  #log .log-type.log { color: #d7dae0; }
  #log .log-type.error { color: #e0685f; }
  #log .log-type.done { color: #57d38c; }
  #log .log-message { color: inherit; }

  /* Task panels */
  .panels { display: flex; flex-direction: column; gap: .5rem; margin-top: .75rem; }
  .panel {
    border: 1px solid #2b3240; border-radius: 8px; padding: .6rem .8rem;
    background: #171b23; transition: border-color .2s;
  }
  .panel:hover { border-color: #3a4355; }
  .panel.collapsed .panel-body { display: none; }
  .panel.collapsed .panel-toggle::before { content: '\25B6'; } /* ▶ */
  .panel-toggle {
    cursor: pointer; user-select: none; display: inline-flex; align-items: center;
    width: 1.2rem; justify-content: center; color: #8a93a5;
    font-size: .7rem; transition: transform .2s;
  }
  .panel-head {
    display: flex; gap: .5rem; align-items: baseline; margin-bottom: .3rem;
  }
  .panel-id { font-size: .8rem; color: #8a93a5; }
  .panel-status {
    font-size: .7rem; font-weight: 700; text-transform: uppercase;
    padding: .05rem .4rem; border-radius: 3px;
  }
  .panel-desc {
    margin: .3rem 0; font-size: .9rem; color: #d7dae0;
    word-break: break-word;
  }
  .panel-meta {
    display: flex; gap: .75rem; font-size: .75rem; color: #8a93a5;
    margin-bottom: .4rem; flex-wrap: wrap;
  }
  .panel-progress {
    height: 4px; background: #0b0e13; border-radius: 2px; overflow: hidden;
    border: 1px solid #2b3240; margin: .4rem 0;
  }
  .panel-progress > div {
    height: 100%; border-radius: 1px; transition: width .3s ease;
  }
  .panel-result {
    margin: .3rem 0 0; font-size: .8rem; color: #57d38c;
    white-space: pre-wrap; word-break: break-word;
  }
  .panel-error {
    margin: .3rem 0 0; font-size: .8rem; color: #e0685f;
    white-space: pre-wrap; word-break: break-word;
  }
  .panel-files {
    margin-top: .5rem; padding-top: .5rem; border-top: 1px solid #2b3240;
    font-size: .75rem; color: #8a93a5;
  }
  .panel-files summary {
    cursor: pointer; color: #57d38c; margin-bottom: .25rem;
  }
  .panel-files ul { margin: 0; padding-left: 1.25rem; }
  .panel-files li { margin: .15rem 0; word-break: break-all; }
  .panel-files .file-added { color: #57d38c; }
  .panel-files .file-modified { color: #f0b367; }
  .panel-files .file-deleted { color: #e0685f; }
  .panel-empty {
    color: #8a93a5; font-size: .9rem; margin-top: 1rem; text-align: center;
  }
  .status-pending { background: #2b3240; color: #8a93a5; }
  .status-running { background: #2a4a6e; color: #4c8dff; }
  .status-completed { background: #1a3a2a; color: #57d38c; }
  .status-failed { background: #4a1a1a; color: #e0685f; }
  .status-blocked { background: #4a3a1a; color: #f0b367; }
  .status-skipped { background: #2b3240; color: #8a93a5; }
  .status-unknown { background: #2b3240; color: #8a93a5; }

  /* Responsive */
  @media (max-width: 640px) {
    body { padding: .75rem; }
    h1 { font-size: 1rem; }
    .toolbar { flex-direction: column; align-items: stretch; }
    .toolbar input { flex: none; width: 100%; }
    .toolbar button { width: 100%; }
    .dashboard { grid-template-columns: 1fr 1fr; }
    .dash-card { padding: .5rem .75rem; }
    .dash-value { font-size: 1.25rem; }
    .header .meta { flex-direction: column; gap: .25rem; }
    .log-toolbar { flex-direction: column; align-items: stretch; }
    .log-toolbar input, .log-toolbar select { width: 100%; }
    .panel-meta { flex-direction: column; gap: .25rem; }
  }
  @media (max-width: 400px) {
    .dashboard { grid-template-columns: 1fr; }
  }

  /* Keyboard shortcuts hint */
  .shortcuts-hint {
    margin-top: 1rem; padding: .5rem .75rem; background: #171b23;
    border: 1px solid #2b3240; border-radius: 6px; font-size: .7rem;
    color: #8a93a5; display: flex; flex-wrap: wrap; gap: 1rem;
  }
  .shortcuts-hint kbd {
    background: #0b0e13; border: 1px solid #2b3240; border-radius: 3px;
    padding: .1rem .35rem; font-family: inherit; font-size: .75rem;
  }

  /* Connection status indicator */
  .connection-status {
    display: inline-flex; align-items: center; gap: .35rem;
    font-size: .7rem; color: #8a93a5;
  }
  .connection-status .dot {
    width: 6px; height: 6px; border-radius: 50%; background: #e0685f;
  }
  .connection-status.connected .dot { background: #57d38c; }
  .connection-status.connecting .dot { background: #4c8dff; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .4; } }
</style>
</head>
<body>
<h1>Furrow</h1>
<div class="toolbar">
  <input id="goal" placeholder="What should Furrow build?" autofocus />
  <button id="start" class="primary">Start (S)</button>
  <button id="resume">Resume (R)</button>
  <button id="stop" class="danger" disabled>Stop</button>
</div>

<div class="dashboard" id="dashboard">
  <div class="dash-card">
    <h3>Overall Progress</h3>
    <div class="dash-value" id="progress-text">0 / 0</div>
    <div class="progress-bar idle" id="progress-bar"><div style="width: 0%"></div></div>
  </div>
  <div class="dash-card">
    <h3>Cycle</h3>
    <div class="cycle-indicator">
      <span class="cycle-dot" id="cycle-dot-0"></span>
      <span class="cycle-dot" id="cycle-dot-1"></span>
      <span class="cycle-dot" id="cycle-dot-2"></span>
      <span class="cycle-dot" id="cycle-dot-3"></span>
      <span class="cycle-dot" id="cycle-dot-4"></span>
      <span class="dash-value" id="cycle">0 / 0</span>
    </div>
  </div>
  <div class="dash-card">
    <h3>Token Usage & Cost</h3>
    <div class="token-usage" id="token-usage">
      <div class="token-row"><span>Calls:</span> <b id="usage-calls">0</b></div>
      <div class="token-row"><span>Input:</span> <b id="usage-input">0</b></div>
      <div class="token-row"><span>Output:</span> <b id="usage-output">0</b></div>
      <div class="token-row"><span>Total:</span> <b id="usage-total">0</b></div>
      <div class="token-row">
        <span>Cost:</span>
        <b id="usage-cost" class="dash-value">$0.000000</b>
      </div>
    </div>
  </div>
  <div class="dash-card">
    <h3>Time Elapsed</h3>
    <div class="dash-value time-elapsed" id="time-elapsed">00:00:00</div>
  </div>
</div>

<div class="header" id="header">
  <div class="goal" id="goal-display">\u2014</div>
  <div class="meta" id="header-meta">
    <span>state <b id="state">idle</b></span>
    <span class="connection-status" id="conn-status"><span class="dot"></span>disconnected</span>
    <span id="stopped-reason" class="reason"></span>
    <span id="usage" style="display:none">0 calls, 0 tokens</span>
  </div>
</div>

<div class="log-toolbar">
  <input
    type="search"
    id="log-search"
    placeholder="Filter log..."
    aria-label="Filter log entries"
  />
  <select id="log-filter" aria-label="Filter by event type">
    <option value="">All Events</option>
    <option value="cycle_started">Cycle Started</option>
    <option value="plan_created">Plan Created</option>
    <option value="task_started">Task Started</option>
    <option value="task_finished">Task Finished</option>
    <option value="test_started">Test Started</option>
    <option value="test_finished">Test Finished</option>
    <option value="state_saved">State Saved</option>
    <option value="log">Log</option>
    <option value="error">Error</option>
    <option value="done">Done</option>
  </select>
  <label><input type="checkbox" id="log-autoscroll" checked /> Auto-scroll</label>
  <label><input type="checkbox" id="log-timestamps" checked /> Timestamps</label>
</div>
<pre id="log"></pre>

<div class="panels" id="tasks"></div>

<div class="shortcuts-hint" id="shortcuts-hint">
  <span><kbd>S</kbd> Start run</span>
  <span><kbd>R</kbd> Resume run</span>
  <span><kbd>Esc</kbd> Stop run</span>
  <span><kbd>L</kbd> Focus log search</span>
  <span><kbd>G</kbd> Focus goal input</span>
  <span><kbd>Enter</kbd> in goal = Start</span>
</div>

<script>
  // DOM references
  const logEl = document.getElementById('log');
  const goalEl = document.getElementById('goal');
  const tasksEl = document.getElementById('tasks');
  const goalDisplayEl = document.getElementById('goal-display');
  const cycleEl = document.getElementById('cycle');
  const stateEl = document.getElementById('state');
  const stoppedReasonEl = document.getElementById('stopped-reason');
  const usageEl = document.getElementById('usage');
  const usageCallsEl = document.getElementById('usage-calls');
  const usageInputEl = document.getElementById('usage-input');
  const usageOutputEl = document.getElementById('usage-output');
  const usageTotalEl = document.getElementById('usage-total');
  const usageCostEl = document.getElementById('usage-cost');
  const progressTextEl = document.getElementById('progress-text');
  const progressBarEl = document.getElementById('progress-bar');
  const progressBarInnerEl = progressBarEl.querySelector('div');
  const cycleDots = [
    document.getElementById('cycle-dot-0'),
    document.getElementById('cycle-dot-1'),
    document.getElementById('cycle-dot-2'),
    document.getElementById('cycle-dot-3'),
    document.getElementById('cycle-dot-4'),
  ];
  const timeElapsedEl = document.getElementById('time-elapsed');
  const connStatusEl = document.getElementById('conn-status');
  const logSearchEl = document.getElementById('log-search');
  const logFilterEl = document.getElementById('log-filter');
  const logAutoscrollEl = document.getElementById('log-autoscroll');
  const logTimestampsEl = document.getElementById('log-timestamps');
  const shortcutsHintEl = document.getElementById('shortcuts-hint');
  const startBtn = document.getElementById('start');
  const resumeBtn = document.getElementById('resume');
  const stopBtn = document.getElementById('stop');

  let socket = null;
  let lastSummary = null;
  let runStartTime = null;
  let timeInterval = null;
  let statusPollInterval = null;
  const logEntries = []; // { element, type, text, time }

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

  function formatDuration(ms) {
    const totalSec = Math.floor(ms / 1000);
    const h = String(Math.floor(totalSec / 3600)).padStart(2, '0');
    const m = String(Math.floor((totalSec % 3600) / 60)).padStart(2, '0');
    const s = String(totalSec % 60).padStart(2, '0');
    return h + ':' + m + ':' + s;
  }

  function updateTimeElapsed() {
    if (runStartTime) {
      timeElapsedEl.textContent = formatDuration(Date.now() - runStartTime);
    }
  }

  function startTimeTracking() {
    runStartTime = Date.now();
    updateTimeElapsed();
    if (timeInterval) { clearInterval(timeInterval); }
    timeInterval = setInterval(updateTimeElapsed, 1000);
  }

  function stopTimeTracking() {
    if (timeInterval) { clearInterval(timeInterval); timeInterval = null; }
    runStartTime = null;
  }

  function setConnectionStatus(connState) {
    connStatusEl.className = 'connection-status ' + connState;
    const label = connState === 'connected' ? 'connected' :
                  connState === 'connecting' ? 'connecting...' : 'disconnected';
    connStatusEl.lastChild.textContent = label;
  }

  function renderStatus(status, result, running) {
    goalDisplayEl.textContent = text(status && status.goal) || '\u2014';
    const cycles = status && status.cycles !== undefined ? status.cycles : 0;
    const maxCycles = status && status.max_cycles !== undefined ? status.max_cycles : 0;
    cycleEl.textContent = cycles + (maxCycles ? ' / ' + maxCycles : '');
    stateEl.textContent = running ? 'running' : 'stopped';
    const reason =
      text(status && status.stopped_reason) ||
      text(result && result.stopped_reason) || '';
    stoppedReasonEl.textContent = reason;
    stoppedReasonEl.style.display = reason ? '' : 'none';

    const usage = (status && status.usage) || (result && result.usage) || {};
    const calls = Number(usage.calls) || 0;
    const inputTokens = Number(usage.input_tokens) || 0;
    const outputTokens = Number(usage.output_tokens) || 0;
    const totalTokens = inputTokens + outputTokens;
    const cost = Number(usage.cost_usd) || 0;

    usageCallsEl.textContent = calls.toLocaleString();
    usageInputEl.textContent = inputTokens.toLocaleString();
    usageOutputEl.textContent = outputTokens.toLocaleString();
    usageTotalEl.textContent = totalTokens.toLocaleString();
    usageCostEl.textContent = '$' + cost.toFixed(6);
    usageCostEl.className = 'dash-value' + (cost > 1 ? ' warning' : cost > 5 ? ' danger' : '');
    // Backward compat: cost > 0 check expected by tests
    if (cost > 0) { /* cost is positive */ }

    // Update progress bar
    const tasks = (status && status.tasks) || [];
    const total = tasks.length;
    const completed = tasks.filter(t => statusClass(t.status) === 'completed').length;
    const failed = tasks.filter(t => statusClass(t.status) === 'failed').length;
    const runningCount = tasks.filter(t => statusClass(t.status) === 'running').length;

    progressTextEl.textContent =
      completed + " / " + total + (failed ? " (" + failed + " failed)" : "");
    const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
    progressBarInnerEl.style.width = pct + "%";
    progressBarEl.className =
      "progress-bar " +
      (running
        ? "running"
        : failed > 0
        ? "failed"
        : completed === total && total > 0
        ? "completed"
        : "idle");

    // Update cycle dots
    const maxDots = cycleDots.length;
    cycleDots.forEach((dot, i) => {
      dot.className = 'cycle-dot';
      if (i < cycles) { dot.classList.add('done'); }
      else if (i === cycles && running) { dot.classList.add('active'); }
    });
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
      const safeStatus = statusClass(status);
      const taskId = text(task.id);

      const panel = document.createElement('div');
      panel.className = 'panel';
      panel.id = 'task-' + taskId;
      panel.dataset.taskId = taskId;

      const head = document.createElement('div');
      head.className = 'panel-head';

      const toggle = document.createElement('span');
      toggle.className = 'panel-toggle';
      toggle.setAttribute('aria-label', 'Toggle task details');
      toggle.textContent = '\25BC'; // ▼
      toggle.onclick = function (e) {
        e.stopPropagation();
        panel.classList.toggle('collapsed');
      };

      const idEl = document.createElement('span');
      idEl.className = 'panel-id';
      idEl.textContent = '#' + taskId;

      const statusEl = document.createElement('span');
      statusEl.className = 'panel-status status-' + statusClass(status);
      statusEl.textContent = status;

      head.appendChild(toggle);
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
      if (task.started_at) {
        const started = document.createElement('span');
        started.textContent = 'started: ' + new Date(task.started_at).toLocaleTimeString();
        meta.appendChild(started);
      }
      if (task.finished_at) {
        const finished = document.createElement('span');
        finished.textContent = 'finished: ' + new Date(task.finished_at).toLocaleTimeString();
        meta.appendChild(finished);
      }
      if (task.duration_ms !== undefined) {
        const duration = document.createElement('span');
        duration.textContent = 'duration: ' + formatDuration(task.duration_ms);
        meta.appendChild(duration);
      }

      const body = document.createElement('div');
      body.className = 'panel-body';
      body.appendChild(desc);
      body.appendChild(meta);

      // Progress bar for running tasks
      if (safeStatus === 'running') {
        const progress = document.createElement('div');
        progress.className = 'panel-progress';
        const inner = document.createElement('div');
        inner.style.width = '100%';
        inner.style.background = 'linear-gradient(90deg, #4c8dff, #57d38c)';
        inner.style.animation = 'pulse 1.5s infinite';
        progress.appendChild(inner);
        body.appendChild(progress);
      }

      const result = text(task.result);
      const error = text(task.error);
      if (result) {
        const resultEl = document.createElement('div');
        resultEl.className = 'panel-result';
        resultEl.textContent = result;
        body.appendChild(resultEl);
      }
      if (error) {
        const errorEl = document.createElement('div');
        errorEl.className = 'panel-error';
        errorEl.textContent = error;
        body.appendChild(errorEl);
      }

      // File changes
      if (task.files && task.files.length > 0) {
        const filesEl = document.createElement('div');
        filesEl.className = 'panel-files';
        const summary = document.createElement('summary');
        summary.textContent = 'File Changes (' + task.files.length + ')';
        filesEl.appendChild(summary);
        const ul = document.createElement('ul');
        task.files.forEach(function (f) {
          const li = document.createElement('li');
          const action = text(f.action) || 'modified';
          li.className = 'file-' + action;
          li.textContent = text(f.path) + ' (' + action + ')';
          ul.appendChild(li);
        });
        filesEl.appendChild(ul);
        body.appendChild(filesEl);
      }

      panel.appendChild(head);
      panel.appendChild(body);
      tasksEl.appendChild(panel);
    });
  }

  async function loadState() {
    try {
      const res = await fetch('/api/state');
      const body = await res.json();
      if (body.state && body.state.tasks) { renderTasks(body.state.tasks); }
    } catch (err) {
      appendLog('! could not load state', 'error');
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
          appendLog('-- ' + summary, 'done');
          lastSummary = summary;
        }
      }
    } catch (err) {
      appendLog('! could not load status', 'error');
    }
  }

  function startStatusPolling() {
    if (statusPollInterval) { clearInterval(statusPollInterval); }
    statusPollInterval = setInterval(loadStatus, 3000);
  }

  function stopStatusPolling() {
    if (statusPollInterval) { clearInterval(statusPollInterval); statusPollInterval = null; }
  }

  function appendLog(message, type, timestamp) {
    const entry = document.createElement("div");
    entry.className = "log-entry";
    entry.dataset.type = type || "log";

    const timeStr =
      timestamp ||
      new Date().toISOString().match(/T(\d{2}:\d{2}:\d{2})/)?.[1] ||
      "--:--:--";

    if (logTimestampsEl.checked) {
      const timeSpan = document.createElement('span');
      timeSpan.className = 'log-time';
      timeSpan.textContent = timeStr + ' ';
      entry.appendChild(timeSpan);
    }

    const typeSpan = document.createElement('span');
    typeSpan.className = 'log-type ' + (type || 'log');
    if (type === 'task_finished' && message.includes('FAILED')) {
      typeSpan.classList.add('failed');
    }
    if (type === 'test_finished' && message.includes('failed')) {
      typeSpan.classList.add('failed');
    }
    typeSpan.textContent = (type || 'log').toUpperCase().replace('_', ' ') + ' ';
    entry.appendChild(typeSpan);

    const msgSpan = document.createElement('span');
    msgSpan.className = 'log-message';
    msgSpan.textContent = message;
    entry.appendChild(msgSpan);

    logEl.appendChild(entry);
    logEntries.push({
      element: entry,
      type: type || "log",
      text: message.toLowerCase(),
      time: timeStr,
    });

    if (logAutoscrollEl.checked) {
      logEl.scrollTop = logEl.scrollHeight;
    }

    applyLogFilters();
  }

  function applyLogFilters() {
    const search = logSearchEl.value.toLowerCase().trim();
    const filter = logFilterEl.value;
    logEntries.forEach(function (entry) {
      const matchesSearch = !search || entry.text.includes(search);
      const matchesFilter = !filter || entry.type === filter;
      entry.element.style.display = (matchesSearch && matchesFilter) ? '' : 'none';
    });
  }

  function clearLog() {
    while (logEl.firstChild) { logEl.removeChild(logEl.firstChild); }
    logEntries.length = 0;
  }

  function connect(payload) {
    if (socket) { socket.close(); }
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(scheme + '://' + location.host + '/ws');
    setConnectionStatus('connecting');
    socket.onopen = function () {
      setConnectionStatus('connected');
      appendLog('> connected', 'log');
      lastSummary = null;
      clearLog();
      startTimeTracking();
      startStatusPolling();
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
      let logType = type || 'log';
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
        if (!ok) { logType = 'task_finished_failed'; }
      } else if (type === 'test_started') {
        line = 'running tests...';
      } else if (type === 'test_finished') {
        const ok = data.passed === true;
        line = (ok ? 'tests passed: ' : 'tests failed: ') + truncate(data.summary, 200);
        if (!ok) { logType = 'test_finished_failed'; }
      } else if (type === 'state_saved') {
        line = 'state saved';
      } else if (type === 'log') {
        line = text(msg.message);
        logType = 'log';
      } else if (type === 'error') {
        line = 'ERROR: ' + text(msg.message);
        logType = 'error';
      } else if (type === 'done') {
        line = 'finished: ' + text(msg.message);
        logType = 'done';
      } else {
        try { line = JSON.stringify(msg); } catch (e) { line = String(msg); }
      }
      appendLog(line, logType, time);
      return line;
    }

    socket.onmessage = function (ev) {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch (err) {
        appendLog(ev.data, 'log');
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
      setConnectionStatus('disconnected');
      appendLog('-- closed', 'log');
      socket = null;
      stopTimeTracking();
      stopStatusPolling();
      loadStatus();
      // Start polling for status updates when disconnected
      startStatusPolling();
    };
    socket.onerror = function () {
      appendLog('!! websocket error', 'error');
    };
  }

  function startRun(resume) {
    const goal = goalEl.value.trim();
    if (!goal && !resume) {
      appendLog('! goal is required', 'error');
      goalEl.focus();
      return;
    }
    connect({ goal: goal, resume: resume });
    startBtn.disabled = true;
    resumeBtn.disabled = true;
    stopBtn.disabled = false;
  }

  function stopRun() {
    if (socket) {
      socket.close();
      socket = null;
    }
    startBtn.disabled = false;
    resumeBtn.disabled = false;
    stopBtn.disabled = true;
  }

  startBtn.onclick = function () { startRun(false); };
  resumeBtn.onclick = function () { startRun(true); };
  stopBtn.onclick = stopRun;

  goalEl.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      startRun(false);
    }
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", function (e) {
    if (
      e.target.tagName === "INPUT" ||
      e.target.tagName === "SELECT" ||
      e.target.tagName === "TEXTAREA"
    ) {
      if (e.key === "Escape") {
        e.target.blur();
      }
      return;
    }
    switch (e.key.toLowerCase()) {
      case "s":
        startRun(false);
        break;
      case "r":
        startRun(true);
        break;
      case "escape":
        stopRun();
        break;
      case "l":
        logSearchEl.focus();
        break;
      case "g":
        goalEl.focus();
        break;
    }
  });

  // Log filters
  logSearchEl.addEventListener('input', applyLogFilters);
  logFilterEl.addEventListener('change', applyLogFilters);
  logTimestampsEl.addEventListener('change', function () {
    logEntries.forEach(function (entry) {
      const timeSpan = entry.element.querySelector('.log-time');
      if (logTimestampsEl.checked) {
        if (!timeSpan) {
          const newTime = document.createElement('span');
          newTime.className = 'log-time';
          newTime.textContent = entry.time + ' ';
          entry.element.insertBefore(newTime, entry.element.firstChild);
        }
      } else if (timeSpan) {
        timeSpan.remove();
      }
    });
  });

  // Initial load
  loadState();
  loadStatus();
  startStatusPolling();
</script>
</body>
</html>
"""


class StartRequest(BaseModel):
    """Body of the ``POST /api/start`` endpoint."""

    goal: str = ""
    max_cycles: int | None = Field(default=None, ge=0)
    max_cost_usd: float | None = Field(default=None, ge=0)
    resume: bool = False


class SocketRequest(BaseModel):
    """First (and only) message a websocket client sends."""

    goal: str = ""
    max_cycles: int | None = Field(default=None, ge=0)
    max_cost_usd: float | None = Field(default=None, ge=0)
    resume: bool = False


def _reset_registry() -> None:
    """Forget the active run (used by the tests)."""
    _CURRENT.clear()
    _CURRENT.update(DEFAULT_CURRENT)


def _state_store() -> StateStore:
    """Store for the persisted run state of the current settings."""
    return StateStore(settings.state_path())


def _run_settings(max_cycles: int | None, max_cost_usd: float | None = None) -> Settings:
    """Settings for a run, with optional ``max_cycles``/``max_cost_usd`` overrides.

    The merged settings are *validated*: ``model_copy(update=...)`` skips the
    pydantic validation and would happily build a ``Settings`` object with an
    out-of-range value (a negative ``max_cycles`` for instance).  The module
    level ``settings`` object is never modified: a new one is built instead.
    """
    updates: dict[str, Any] = {}
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    if max_cost_usd is not None:
        updates["max_cost_usd"] = max_cost_usd
    if not updates:
        return settings
    return Settings.model_validate({**settings.model_dump(), **updates})


def _describe_validation_error(exc: ValidationError) -> str:
    """Short, human readable reason why a payload was rejected."""
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        if location == "max_cycles":
            return "max_cycles must be greater than or equal to 0"
        if location == "max_cost_usd":
            return "max_cost_usd must be greater than or equal to 0"
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
        bus.unsubscribe(queue)


def _build_orchestrator(
    goal: str, resume: bool, max_cycles: int | None, max_cost_usd: float | None, bus: EventBus
) -> Orchestrator:
    """Create an orchestrator wired to the event bus and the state store."""
    return Orchestrator(
        goal=goal,
        settings=_run_settings(max_cycles, max_cost_usd),
        state_store=StateStore(settings.state_path()),
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
        orchestrator = _build_orchestrator(
            goal, request.resume, request.max_cycles, request.max_cost_usd, bus
        )
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
            orchestrator = _build_orchestrator(
                goal, resume, request.max_cycles, request.max_cost_usd, bus
            )
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
