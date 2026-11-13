/**
 * Furrow Web UI - Main Application
 * Handles WebSocket communication, event rendering, and UI updates
 */

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

let socket = null;
let lastSummary = null;

// Safe status mapping to prevent XSS
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
  const key = String(status || '').trim().toLowerCase();
  return SAFE_STATUSES[key] || 'unknown';
}

function text(value) {
  return value === undefined || value === null ? '' : String(value);
}

function truncate(value, max) {
  const s = text(value);
  return s.length > max ? s.slice(0, max) + '…' : s;
}

function eventTime(msg) {
  const ts = msg && msg.timestamp;
  if (!ts) { return '--:--:--'; }
  const m = String(ts).match(/T(\d{2}:\d{2}:\d{2})/);
  return m ? m[1] : '--:--:--';
}

function appendLog(line, color = '#8a93a5') {
  const span = document.createElement('span');
  span.style.color = color;
  span.textContent = line;
  logEl.appendChild(span);
  logEl.appendChild(document.createTextNode('\n'));
  logEl.scrollTop = logEl.scrollHeight;
}

function renderStatus(status, result, running) {
  goalDisplayEl.textContent = text(status && status.goal) || '—';
  cycleEl.textContent = text(status && status.cycles !== undefined ? status.cycles : '—');
  stateEl.textContent = running ? 'running' : 'stopped';

  const reason = text(status && status.stopped_reason) || text(result && result.stopped_reason) || '';
  stoppedReasonEl.textContent = reason;
  stoppedReasonEl.style.display = reason ? '' : 'none';

  const usage = (status && status.usage) || (result && result.usage) || {};
  const calls = Number(usage.calls) || 0;
  const total = (Number(usage.input_tokens) || 0) + (Number(usage.output_tokens) || 0);
  const cost = Number(usage.cost_usd) || 0;

  usageEl.textContent = `${calls} calls, ${total} tokens` + (cost > 0 ? `, $${cost.toFixed(6)}` : '');
}

function renderTasks(tasks) {
  while (tasksEl.firstChild) {
    tasksEl.removeChild(tasksEl.firstChild);
  }

  const list = tasks || [];

  if (list.length === 0) {
    const hint = document.createElement('p');
    hint.className = 'panel-empty';
    hint.textContent = 'No run yet — enter a goal and click Start.';
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

function renderEvent(msg) {
  const type = msg && msg.type;
  const data = (msg && msg.data) || {};
  const time = eventTime(msg);

  let line = '';
  let color = '#8a93a5';

  switch (type) {
    case 'cycle_started':
      line = `cycle ${text(data.cycle)} started`;
      color = '#4c8dff';
      break;
    case 'plan_created':
      line = `plan: ${truncate(data.rationale, 200)} (${text(data.tasks)} tasks)`;
      color = '#4c8dff';
      break;
    case 'task_started':
      line = `task ${text(data.task)} started`;
      color = '#4c8dff';
      break;
    case 'task_finished': {
      const ok = data.success === true;
      line = `task ${text(data.task)} ${ok ? 'completed: ' : 'FAILED: '}${truncate(data.summary, 200)}`;
      color = ok ? '#57d38c' : '#e0685f';
      break;
    }
    case 'test_started':
      line = 'running tests...';
      color = '#f0b367';
      break;
    case 'test_finished': {
      const ok = data.passed === true;
      line = `${ok ? 'tests passed: ' : 'tests failed: '}${truncate(data.summary, 200)}`;
      color = ok ? '#57d38c' : '#e0685f';
      break;
    }
    case 'state_saved':
      line = 'state saved';
      color = '#8a93a5';
      break;
    case 'log':
      line = text(msg.message);
      color = '#d7dae0';
      break;
    case 'error':
      line = 'ERROR: ' + text(msg.message);
      color = '#e0685f';
      break;
    case 'done':
      line = 'finished: ' + text(msg.message);
      color = '#57d38c';
      break;
    default:
      try { line = JSON.stringify(msg); } catch (e) { line = String(msg); }
  }

  appendLog(`${time} ${line}`, color);
  return line;
}

async function loadState() {
  try {
    const res = await fetch('/api/state');
    const body = await res.json();
    if (body.state && body.state.tasks) {
      renderTasks(body.state.tasks);
    }
  } catch (err) {
    appendLog('! could not load state', '#e0685f');
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
    }

    if (body.result) {
      const summary = text(body.result.summary);
      if (summary && summary !== lastSummary) {
        appendLog('-- ' + summary, '#57d38c');
        lastSummary = summary;
      }
    }
  } catch (err) {
    appendLog('! could not load status', '#e0685f');
  }
}

function connect(payload) {
  if (socket) {
    socket.close();
  }

  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${scheme}://${location.host}/ws`);

  socket.onopen = function () {
    appendLog('> connected', '#57d38c');
    lastSummary = null;
    socket.send(JSON.stringify(payload));
    startBtn.disabled = true;
    resumeBtn.disabled = true;
  };

  socket.onmessage = function (ev) {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch (err) {
      appendLog(ev.data);
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
    }
  };

  socket.onclose = function () {
    appendLog('-- closed', '#8a93a5');
    socket = null;
    startBtn.disabled = false;
    resumeBtn.disabled = false;
    loadStatus();
  };

  socket.onerror = function () {
    appendLog('!! websocket error', '#e0685f');
  };
}

startBtn.onclick = function () {
  connect({ goal: goalEl.value, resume: false });
};

resumeBtn.onclick = function () {
  connect({ goal: goalEl.value, resume: true });
};

// Load initial state on page load
loadState();
loadStatus();

// Allow Enter key in goal input to start
goalEl.addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (goalEl.value.trim()) {
      connect({ goal: goalEl.value, resume: false });
    }
  }
});