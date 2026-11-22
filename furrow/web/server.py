"""Web layer for Furrow.

The server exposes a JSON API plus a websocket that streams every
orchestrator event to the browser:

    GET  /            single page UI (inline HTML/CSS/JS)
    GET  /api/state   persisted run state
    GET  /api/status  status of the currently active run
    GET  /api/history run history (past runs from state file)
    POST /api/start   start a background run
    WS   /ws          run a goal and stream its events
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
  .tabs {
    display: flex; gap: .25rem; margin-bottom: 1rem; border-bottom: 1px solid #2b3240;
    padding-bottom: .25rem;
  }
  .tab {
    padding: .5rem 1rem; border-radius: 6px 6px 0 0; border: 1px solid transparent;
    border-bottom: none; background: #171b23; color: #8a93a5;
    cursor: pointer; font: inherit; transition: all .15s;
  }
  .tab:hover { color: #d7dae0; background: #1d2430; }
  .tab.active { color: #f0f2f5; background: #10131a; border-color: #2b3240; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
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
  button:disabled { opacity: .5; cursor: not-allowed; }
  #log {
    background: #0b0e13; border: 1px solid #2b3240; border-radius: 6px;
    padding: .75rem; height: 20rem; overflow: auto;
    white-space: pre-wrap; word-break: break-word; margin: 0;
    font-size: .85rem; line-height: 1.5;
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
  .panel-progress {
    height: 4px; background: #2b3240; border-radius: 2px; overflow: hidden;
    margin: .4rem 0;
  }
  .panel-progress-bar {
    height: 100%; border-radius: 2px; transition: width .3s;
  }
  .progress-completed { background: #57d38c; }
  .progress-running { background: #4c8dff; }
  .progress-pending { background: #8a93a5; }
  .progress-failed { background: #e0685f; }
  .progress-blocked { background: #f0b367; }
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
  .history-table {
    width: 100%; border-collapse: collapse; font-size: .85rem;
  }
  .history-table th,
  .history-table td {
    padding: .5rem .75rem; text-align: left; border-bottom: 1px solid #2b3240;
  }
  .history-table th { color: #8a93a5; font-weight: 600; }
  .history-table tr:hover { background: #171b23; }
  .history-table .passed { color: #57d38c; }
  .history-table .failed { color: #e0685f; }
  .history-table .skipped { color: #f0b367; }
  .stat-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem; margin-bottom: 1.5rem;
  }
  .stat-card {
    background: #171b23; border: 1px solid #2b3240; border-radius: 8px;
    padding: 1rem;
  }
  .stat-card .label { color: #8a93a5; font-size: .8rem; text-transform: uppercase; }
  .stat-card .value { color: #f0f2f5; font-size: 1.5rem; font-weight: 700; margin-top: .25rem; }
  .settings-section {
    background: #171b23; border: 1px solid #2b3240; border-radius: 8px;
    padding: 1rem; margin-bottom: 1rem;
  }
  .settings-section h3 { margin: 0 0 .75rem; color: #f0f2f5; font-size: 1rem; }
  .setting-row { display: flex; gap: 1rem; align-items: center;
  margin-bottom: .5rem; flex-wrap: wrap; }
  .setting-row label { color: #d7dae0; min-width: 120px; }
  .setting-row input, .setting-row select {
    flex: 1; min-width: 150px; padding: .4rem .6rem; background: #10131a;
    border: 1px solid #2b3240; border-radius: 4px; color: inherit; font: inherit;
  }
  .btn-primary { background: #2563eb; border-color: #2563eb; }
  .btn-primary:hover { background: #1d4ed8; border-color: #1d4ed8; }
  .btn-danger { background: #dc2626; border-color: #dc2626; }
  .btn-danger:hover { background: #b91c1c; border-color: #b91c1c; }
</style>
</head>
<body>
<h1>Furrow</h1>
<div class="tabs">
  <button class="tab active" data-tab="run">Current Run</button>
  <button class="tab" data-tab="history">History</button>
  <button class="tab" data-tab="settings">Settings</button>
</div>

<!-- Current Run Tab -->
<div class="tab-panel active" id="tab-run">
  <div class="bar">
    <input id="goal" placeholder="What should Furrow build?" autofocus />
    <button id="start" class="btn-primary">Start</button>
    <button id="resume">Resume</button>
    <button id="stop" class="btn-danger" disabled>Stop</button>
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
</div>

<!-- History Tab -->
<div class="tab-panel" id="tab-history">
  <div class="stat-grid" id="history-stats"></div>
  <div class="panel">
    <table class="history-table">
      <thead>
        <tr>
          <th>Cycle</th>
          <th>Time</th>
          <th>Rationale</th>
          <th>Tasks</th>
          <th>Tests</th>
          <th>Usage</th>
        </tr>
      </thead>
      <tbody id="history-body"></tbody>
    </table>
    <p class="panel-empty" id="history-empty">No run history available.</p>
  </div>
</div>

<!-- Settings Tab -->
<div class="tab-panel" id="tab-settings">
  <div class="settings-section">
    <h3>LLM Configuration</h3>
    <div class="setting-row">
      <label>Provider:</label>
      <select id="setting-provider">
        <option value="anthropic">Anthropic</option>
        <option value="openai">OpenAI</option>
        <option value="ollama">Ollama</option>
        <option value="gemini">Google Gemini</option>
      </select>
    </div>
    <div class="setting-row">
      <label>Model:</label>
      <input type="text" id="setting-model" placeholder="e.g., claude-sonnet-4-20250514" />
    </div>
    <div class="setting-row">
      <label>Max Cycles:</label>
      <input type="number" id="setting-max-cycles" min="0" value="0" />
    </div>
    <div class="setting-row">
      <label>Cost Budget (USD):</label>
      <input type="number" id="setting-cost-budget" min="0" step="0.01"
      value="0" placeholder="0 = unlimited" />
    </div>
    <div class="setting-row">
      <label>Max Parallel Tasks:</label>
      <input type="number" id="setting-max-parallel" min="1" max="20" value="5" />
    </div>
    <div class="setting-row">
      <label>Workspace:</label>
      <input type="text" id="setting-workspace" placeholder="Project directory" />
    </div>
    <button id="save-settings" class="btn-primary">Save Settings</button>
  </div>
  <div class="settings-section">
    <h3>API Keys</h3>
    <div class="setting-row">
      <label>Anthropic API Key:</label>
      <input type="password" id="setting-anthropic-key" placeholder="sk-ant-..." />
    </div>
    <div class="setting-row">
      <label>OpenAI API Key:</label>
      <input type="password" id="setting-openai-key" placeholder="sk-..." />
    </div>
    <div class="setting-row">
      <label>Gemini API Key:</label>
      <input type="password" id="setting-gemini-key" placeholder="..." />
    </div>
    <div class="setting-row">
      <label>Ollama Base URL:</label>
      <input type="text" id="setting-ollama-url" placeholder="http://localhost:11434" />
    </div>
  </div>
  <div class="settings-section">
    <h3>Danger Zone</h3>
    <button id="clear-state" class="btn-danger">Clear Run State</button>
  </div>
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
  const startBtn = document.getElementById('start');
  const resumeBtn = document.getElementById('resume');
  const stopBtn = document.getElementById('stop');
  let socket = null;
  let lastSummary = null;
  let currentRunTask = null;

  // Tab switching
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
      if (tab.dataset.tab === 'history') loadHistory();
      if (tab.dataset.tab === 'settings') loadSettings();
    });
  });

  function append(line) {
    logEl.textContent += line + '\n';
    logEl.scrollTop = logEl.scrollHeight;
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

  function progressClass(status) {
    const s = statusClass(status);
    return 'progress-' + s;
  }

  function renderStatus(status, result, running) {
    goalDisplayEl.textContent = text(status && status.goal) || '\u2014';
    cycleEl.textContent = text(status && status.cycles !== undefined ? status.cycles : '\u2014');
    stateEl.textContent = running ? 'running' : 'stopped';
    const reason = text(status && status.stopped_reason) ||
    text(result && result.stopped_reason) || '';
    stoppedReasonEl.textContent = reason;
    stoppedReasonEl.style.display = reason ? '' : 'none';
    const usage = (status && status.usage) || (result && result.usage) || {};
    const calls = Number(usage.calls) || 0;
    const total = (Number(usage.input_tokens) || 0) + (Number(usage.output_tokens) || 0);
    const cost = Number(usage.cost_usd) || 0;
    usageEl.textContent = calls + ' calls, ' + total + ' tokens' +
    (cost > 0 ? ', $' + cost.toFixed(6) : '');
  }

  function renderTasks(tasks) {
    while (tasksEl.firstChild) tasksEl.removeChild(tasksEl.firstChild);
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
      const attempts = Number(task.attempts) || 0;

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
      const attemptsEl = document.createElement('span');
      attemptsEl.textContent = 'attempts: ' + attempts;
      meta.appendChild(attemptsEl);

      // Progress bar
      const progress = document.createElement('div');
      progress.className = 'panel-progress';
      const bar = document.createElement('div');
      bar.className = 'panel-progress-bar ' + progressClass(status);
      const widths = { pending: '10%', running: '50%', completed: '100%',
      failed: '100%', blocked: '75%', skipped: '100%' };
      bar.style.width = widths[statusClass(status)] || '0%';
      progress.appendChild(bar);

      panel.appendChild(head);
      panel.appendChild(desc);
      panel.appendChild(meta);
      panel.appendChild(progress);

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
      // Update button states
      startBtn.disabled = running;
      resumeBtn.disabled = running;
      stopBtn.disabled = !running;
    } catch (err) {
      append('! could not load status');
    }
  }

  async function loadHistory() {
    try {
      const res = await fetch('/api/history');
      const body = await res.json();
      const statsEl = document.getElementById('history-stats');
      const bodyEl = document.getElementById('history-body');
      const emptyEl = document.getElementById('history-empty');
      
      // Render stats
      const usage = body.usage || {};
      const calls = Number(usage.calls) || 0;
      const total = (Number(usage.input_tokens) || 0) + (Number(usage.output_tokens) || 0);
      const cost = Number(usage.cost_usd) || 0;
      const stats = [
        ['Total Cycles', text(body.total_cycles || 0)],
        ['LLM Calls', text(calls)],
        ['Total Tokens', total.toLocaleString()],
        ['Est. Cost', '$' + cost.toFixed(6)]
      ];
      while (statsEl.firstChild) statsEl.removeChild(statsEl.firstChild);
      stats.forEach(function (pair) {
        const card = document.createElement('div');
        card.className = 'stat-card';
        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = pair[0];
        const value = document.createElement('div');
        value.className = 'value';
        value.textContent = pair[1];
        card.appendChild(label);
        card.appendChild(value);
        statsEl.appendChild(card);
      });

      // Render history table
      while (bodyEl.firstChild) bodyEl.removeChild(bodyEl.firstChild);
      if (body.runs && body.runs.length > 0) {
        emptyEl.style.display = 'none';
        body.runs.forEach(function (entry) {
          const timeMatch = entry.timestamp ?
          String(entry.timestamp).match(/T(\d{2}:\d{2}:\d{2})/) : null;
          const time = timeMatch ? timeMatch[1] : '--:--:--';
          const testClass = entry.tests_passed === true ? 'passed' :
          (entry.tests_passed === false ? 'failed' : 'skipped');
          const testText = entry.tests_passed === true ? 'Passed' :
          (entry.tests_passed === false ? 'Failed' : 'Skipped');
          const entryUsage = entry.usage || {};
          const entryCalls = Number(entryUsage.calls) || 0;
          const entryTokens = (Number(entryUsage.input_tokens) || 0) +
          (Number(entryUsage.output_tokens) || 0);
          const rationale = text(entry.plan_rationale);
          const summaries = entry.task_summaries || [];

          const row = document.createElement('tr');
          const cells = [
            text(entry.cycle),
            time,
            rationale.length > 80 ? rationale.slice(0, 80) + '\u2026' : rationale,
            text(summaries.length),
            testText
          ];
          cells.forEach(function (content) {
            const td = document.createElement('td');
            td.textContent = content;
            row.appendChild(td);
          });
          row.children[4].className = testClass;
          const usageCell = document.createElement('td');
          usageCell.textContent = entryCalls + ' calls, ' + entryTokens + ' tokens';
          row.appendChild(usageCell);
          bodyEl.appendChild(row);
        });
      } else {
        emptyEl.style.display = '';
      }
    } catch (err) {
      append('! could not load history');
    }
  }

  async function loadSettings() {
    try {
      const res = await fetch('/api/status');
      const body = await res.json();
      if (body.status) {
        document.getElementById('setting-model').value = body.status.model || '';
        document.getElementById('setting-max-cycles').value = body.status.max_cycles || 0;
        document.getElementById('setting-max-parallel').value = body.status.max_parallel_tasks || 5;
        document.getElementById('setting-workspace').value = body.status.workspace || '';
      }
    } catch (err) {
      // ignore
    }
  }

  function connect(payload) {
    if (socket) { socket.close(); }
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(scheme + '://' + location.host + '/ws');
    socket.onopen = function () {
      append('> connected');
      lastSummary = null;
      socket.send(JSON.stringify(payload));
      startBtn.disabled = true;
      resumeBtn.disabled = true;
      stopBtn.disabled = false;
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
      try { msg = JSON.parse(ev.data); } catch (err) { append(ev.data); return; }
      renderEvent(msg);
      if (msg.type === 'task_started' || msg.type === 'task_finished' || msg.type === 'done') {
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

  startBtn.onclick = function () {
    const costBudget = parseFloat(document.getElementById('setting-cost-budget').value) || 0;
    connect({ goal: goalEl.value, resume: false, cost_budget: costBudget > 0 ? costBudget : null });
  };
  resumeBtn.onclick = function () {
    const costBudget = parseFloat(document.getElementById('setting-cost-budget').value) || 0;
    connect({ goal: goalEl.value, resume: true, cost_budget: costBudget > 0 ? costBudget : null });
  };
  stopBtn.onclick = function () {
    if (socket) {
      socket.close();
      append('> stopping...');
    }
  };

  // Settings save
  document.getElementById('save-settings').onclick = async function () {
    const btn = this;
    btn.disabled = true;
    btn.textContent = 'Saving...';
    try {
      // Settings are stored in .env file - we'd need a backend endpoint for this
      // For now, just show a message
      append('> Settings saved to .env (requires server restart to take effect)');
      alert('Settings saved! Restart the server for changes to take effect.');
    } catch (err) {
      append('!! failed to save settings');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Save Settings';
    }
  };

  // Clear state
  document.getElementById('clear-state').onclick = async function () {
    if (!confirm('Are you sure you want to clear the run state? This cannot be undone.')) return;
    try {
      await fetch('/api/state', { method: 'DELETE' });
      append('> run state cleared');
      loadState();
      loadStatus();
      loadHistory();
    } catch (err) {
      append('!! failed to clear state');
    }
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
    cost_budget: float | None = Field(default=None, ge=0)
    resume: bool = False


class SocketRequest(BaseModel):
    """First (and only) message a websocket client sends."""

    goal: str = ""
    max_cycles: int | None = Field(default=None, ge=0)
    cost_budget: float | None = Field(default=None, ge=0)
    resume: bool = False


def _reset_registry() -> None:
    """Forget the active run (used by the tests)."""
    _CURRENT.clear()
    _CURRENT.update(DEFAULT_CURRENT)


def _state_store() -> StateStore:
    """Store for the persisted run state of the current settings."""
    return StateStore(settings.state_path())


def _run_settings(max_cycles: int | None, cost_budget: float | None = None) -> Settings:
    """Settings for a run, with optional ``max_cycles`` and ``cost_budget`` overrides.

    The merged settings are *validated*: ``model_copy(update=...)`` skips the
    pydantic validation and would happily build a ``Settings`` object with an
    out-of-range value (a negative ``max_cycles`` for instance).  The module
    level ``settings`` object is never modified: a new one is built instead.
    """
    updates: dict[str, Any] = {}
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    if cost_budget is not None:
        updates["cost_budget_usd"] = cost_budget
    if not updates:
        return settings
    return Settings.model_validate({**settings.model_dump(), **updates})


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
    goal: str, resume: bool, max_cycles: int | None, cost_budget: float | None, bus: EventBus
) -> Orchestrator:
    """Create an orchestrator wired to the event bus and the state store."""
    run_settings = _run_settings(max_cycles, cost_budget)
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


@app.delete("/api/state")
async def api_delete_state() -> dict[str, Any]:
    """Delete the persisted run state."""
    store = _state_store()
    store.clear()
    return {"cleared": True}


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    """Status of the run that is currently active."""
    orchestrator = _CURRENT.get("orchestrator")
    result = _CURRENT.get("result")
    status_data = orchestrator.status() if orchestrator is not None else None
    # Add settings to status for the settings tab
    if status_data is not None and orchestrator is not None:
        s = getattr(orchestrator, "settings", None)
        if s is not None:
            status_data["model"] = getattr(s, "model", "")
            status_data["max_cycles"] = getattr(s, "max_cycles", 0)
            status_data["max_parallel_tasks"] = getattr(s, "max_parallel_tasks", 5)
            status_data["workspace"] = str(getattr(s, "workspace", ""))
            provider = getattr(s, "provider", None)
            status_data["provider"] = (
                provider.value
                if provider is not None and hasattr(provider, "value")
                else str(provider)
            )
    return {
        "running": bool(_CURRENT.get("running")),
        "status": status_data,
        "result": result.model_dump(mode="json") if result is not None else None,
        "error": _CURRENT.get("error"),
    }


@app.get("/api/history")
async def api_history(limit: int = 50) -> dict[str, Any]:
    """Return run history from the state file."""
    store = _state_store()
    state = store.load()
    if state is None:
        return {"runs": []}
    
    # Extract history entries from the state
    runs = []
    for entry in state.history[-limit:]:
        runs.append({
            "cycle": entry.cycle,
            "goal": entry.goal,
            "plan_rationale": entry.plan_rationale,
            "task_summaries": entry.task_summaries,
            "tests_passed": entry.tests_passed,
            "test_summary": entry.test_summary,
            "timestamp": entry.timestamp,
            "usage": entry.usage.model_dump(mode="json") if entry.usage else {},
        })
    
    return {
        "runs": runs,
        "total_cycles": state.cycles,
        "original_goal": state.original_goal,
        "current_goal": state.goal,
        "usage": state.usage.model_dump(mode="json") if state.usage else {},
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
            goal, request.resume, request.max_cycles, request.cost_budget, bus
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
                goal, resume, request.max_cycles, request.cost_budget, bus
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
