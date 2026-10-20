from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import settings
from furrow.core.events import Event
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import save_state, state_from_orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


HTML = """<!DOCTYPE html>
<html>
<head>
<title>Furrow</title>
<style>
body { font-family: system-ui, sans-serif; padding: 16px; }
#bar { display: flex; gap: 8px; align-items: center; margin-bottom: 16px; }
#log { border: 1px solid #ccc; padding: 8px; height: 300px; overflow: auto;
  background: #fafafa; font-family: ui-monospace, monospace; font-size: 13px; }
#log div { white-space: pre-wrap; word-break: break-word; margin: 2px 0; }
#tasks { border: 1px solid #ccc; padding: 8px; }
.task { margin: 4px 0; display: flex; gap: 8px; }
.kind { color: #666; }
.ok { color: green; }
.bad { color: red; }
#status { margin-bottom: 8px; font-weight: bold; }
.btn { cursor: pointer; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.hidden { display: none; }
</style>
</head>
<body>
<h1>Furrow</h1>
<div id="status"><span id="statusDot">?</span> <span id="statusText">disconnected</span></div>
<div id="bar">
  <input id="goal" placeholder="Enter goal" />
  <button id="startBtn" class="btn">Start</button>
  <button id="stopBtn" class="btn">Stop</button>
  <button id="reconnectBtn" class="btn hidden">Reconnect</button>
</div>
<div id="log"></div>
<h3>Tasks</h3>
<div id="tasks"></div>
<script>
const goalInput = document.getElementById('goal');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const reconnectBtn = document.getElementById('reconnectBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const log = document.getElementById('log');
const taskList = document.getElementById('tasks');
let ws = null;
let tasks = {};

function setConnected(open) {
  if (open) {
    statusDot.textContent = 'connected';
    statusDot.className = 'ok';
    statusText.textContent = 'connected';
    reconnectBtn.classList.add('hidden');
  } else {
    statusDot.textContent = 'disconnected';
    statusDot.className = 'bad';
    statusText.textContent = 'disconnected';
    reconnectBtn.classList.remove('hidden');
  }
}

function appendLine(text) {
  const div = document.createElement('div');
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function eventText(kind, data) {
  const parts = ['[' + String(kind) + ']'];
  switch (kind) {
    case 'cycle_start':
      parts.push('cycle ' + String(data == null ? '' : data.cycle));
      break;
    case 'plan': {
      const n = Array.isArray(data == null ? null : data.tasks) ? data.tasks.length : 0;
      parts.push('planned ' + String(n) + ' task(s)');
      if (data != null && data.rationale) parts.push('rationale: ' + String(data.rationale));
      break;
    }
    case 'task_start':
      parts.push('task ' + String(data == null ? '' : data.task_id)
        + ' started: ' + String(data == null ? '' : data.description ?? ''));
      break;
    case 'task_done':
      parts.push('task ' + String(data == null ? '' : data.task_id)
        + ' done: ' + String(data == null ? '' : data.result ?? ''));
      break;
    case 'task_failed':
      parts.push('task ' + String(data == null ? '' : data.task_id)
        + ' failed: ' + String(data == null ? '' : data.error ?? ''));
      break;
    case 'test_result':
      parts.push('tests ' + (data != null && data.passed ? 'passed' : 'failed')
        + ': ' + String(data == null ? '' : data.summary ?? ''));
      break;
    case 'cycle_end':
      parts.push('cycle ' + String(data == null ? '' : data.cycle) + ' end');
      break;
    case 'done':
      parts.push('done: ' + String(data == null ? '' : data.reason)
        + ' (' + String(data == null ? '' : data.cycles) + ' cycles)');
      break;
    default:
      parts.push(JSON.stringify(data));
  }
  return parts.join(' ');
}

function handleEvent(event) {
  if (event == null || typeof event !== 'object') return;
  const kind = String(event.kind ?? '');
  const data = event.data ?? {};
  appendLine(eventText(kind, data));
  if (kind === 'plan' && Array.isArray(data.tasks)) {
    for (const t of data.tasks) {
      tasks[t.id] = { id: t.id, description: t.description, status: 'pending' };
    }
  } else if (kind === 'task_start') {
    const t = tasks[data.task_id] || { id: data.task_id, description: '', status: '' };
    t.status = 'in_progress';
    tasks[data.task_id] = t;
  } else if (kind === 'task_done') {
    const t = tasks[data.task_id] || { id: data.task_id, description: '', status: '' };
    t.status = 'completed';
    tasks[data.task_id] = t;
  } else if (kind === 'task_failed') {
    const t = tasks[data.task_id] || { id: data.task_id, description: '', status: '' };
    t.status = 'failed';
    tasks[data.task_id] = t;
  } else if (kind === 'done') {
    refreshState();
  }
  renderTasks();
}

function renderTasks() {
  taskList.textContent = '';
  for (const id of Object.keys(tasks)) {
    const t = tasks[id];
    const div = document.createElement('div');
    div.className = 'task';
    div.textContent = t.id + ': ' + t.description + ' — ' + t.status;
    taskList.appendChild(div);
  }
}

async function refreshState() {
  try {
    const res = await fetch('/api/state');
    const state = await res.json();
    if (state != null && Array.isArray(state.tasks)) {
      tasks = {};
      for (const t of state.tasks) {
        tasks[t.id] = { id: t.id, description: t.description, status: t.status };
      }
      renderTasks();
    }
  } catch (e) {
    // ignore
  }
}

function connect() {
  if (ws != null && ws.readyState === WebSocket.OPEN) return;
  ws = new WebSocket('ws://' + location.host + '/ws');
  ws.onopen = () => setConnected(true);
  ws.onmessage = (ev) => {
    try { handleEvent(JSON.parse(ev.data)); } catch (e) { /* ignore */ }
  };
  ws.onclose = () => { setConnected(false); ws = null; };
  ws.onerror = () => { ws.close(); };
}

async function start() {
  const goal = goalInput.value.trim();
  if (!goal) return;
  try {
    await fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ goal }),
    });
  } catch (e) { /* 409 if already running; events still stream */ }
  connect();
}

async function stop() {
  try { await fetch('/api/stop', { method: 'POST' }); } catch (e) { /* ignore */ }
}

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
reconnectBtn.addEventListener('click', () => { connect(); });
connect();
</script>
</body>
</html>
"""


class _Subscriber:
    __slots__ = ("closed", "queue", "sender")

    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(
            maxsize=1000
        )
        self.sender: asyncio.Task[None] | None = None
        self.closed = False


class RunManager:
    def __init__(self) -> None:
        self._orchestrator: Orchestrator | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._subscribers: dict[str, _Subscriber] = {}
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def orchestrator(self) -> Orchestrator | None:
        return self._orchestrator

    def _on_event(self, event: Event) -> None:
        payload = event.to_dict()
        for sub in list(self._subscribers.values()):
            try:
                sub.queue.put_nowait(payload)
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    sub.queue.get_nowait()
                sub.queue.put_nowait(payload)
        if event.kind in ("cycle_end", "done"):
            try:
                orch = self._orchestrator
                if orch is not None:
                    st = state_from_orchestrator(orch)
                    save_state(st, settings.state_file)
            except Exception:
                logging.getLogger(__name__).exception("failed to persist run state")

    async def _pump(self, client_id: str, websocket: WebSocket) -> None:
        sub = self._subscribers.get(client_id)
        if sub is None:
            return
        try:
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                try:
                    await websocket.send_json(item)
                except Exception:
                    return
        finally:
            await self.unsubscribe(client_id)

    async def subscribe(self, client_id: str, websocket: WebSocket) -> None:
        sub = _Subscriber()
        sub.sender = asyncio.create_task(self._pump(client_id, websocket))
        async with self._lock:
            self._subscribers[client_id] = sub

    async def unsubscribe(self, client_id: str) -> None:
        async with self._lock:
            sub = self._subscribers.pop(client_id, None)
        if sub is None or sub.closed:
            return
        sub.closed = True
        if sub.sender is not None:
            sub.sender.cancel()

    async def _run(self, goal: str) -> None:
        self._stop.clear()
        self._orchestrator = Orchestrator(goal=goal, on_event=self._on_event)
        try:
            await self._orchestrator.run()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            self._task = None
            self._stop.set()

    async def start(self, goal: str) -> None:
        if self.running:
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(goal))

    async def stop(self) -> bool:
        if not self.running:
            return False
        self._stop.set()
        task = self._task
        if task is not None:
            task.cancel()
        return True


manager = RunManager()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    client_id = str(uuid.uuid4())
    await manager.subscribe(client_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            goal = ""
            if isinstance(data, dict):
                goal = str(data.get("goal", ""))
            if goal and not manager.running:
                await manager.start(goal)
    except Exception:
        pass
    finally:
        await manager.unsubscribe(client_id)


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "running": manager.running}


@app.get("/api/state")
async def state() -> dict[str, object]:
    orchestrator = manager.orchestrator
    if orchestrator is None:
        return {"goal": "", "cycles": 0, "tasks": [], "history": []}
    task_entries: list[dict[str, object]] = []
    for task in orchestrator.tasks:
        dumped = task.model_dump(mode="json")
        task_entries.append(
            {
                "id": dumped["id"],
                "description": dumped["description"],
                "status": dumped["status"],
                "files": dumped["files"],
                "files_changed": getattr(task, "files_changed", None) or [],
            }
        )
    return {
        "goal": orchestrator.goal,
        "cycles": orchestrator.cycles,
        "tasks": task_entries,
        "history": list(orchestrator.history),
    }


@app.post("/api/start", status_code=status.HTTP_202_ACCEPTED)
async def start_run(request: StartRequest) -> dict[str, object]:
    if manager.running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="a run is already in progress"
        )
    await manager.start(request.goal)
    return {"started": True, "goal": request.goal}


@app.api_route("/api/stop", methods=["GET", "POST"])
async def stop_run() -> dict[str, object]:
    return {"stopped": await manager.stop()}


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
