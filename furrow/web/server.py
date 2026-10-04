from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.config import settings as default_settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    workspace: str | None = None


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Furrow</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 24px;
    background: #0f172a;
    color: #e2e8f0;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  h1 { margin: 0 0 4px; font-size: 22px; letter-spacing: 0.08em; }
  p.sub { margin: 0 0 20px; color: #64748b; font-size: 13px; }
  form { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }
  input {
    flex: 1 1 320px;
    padding: 10px 12px;
    border-radius: 6px;
    border: 1px solid #334155;
    background: #111c33;
    color: #e2e8f0;
    font: inherit;
  }
  input.ws { flex: 0 1 260px; }
  input:focus { outline: 1px solid #38bdf8; border-color: #38bdf8; }
  button {
    padding: 10px 20px;
    border-radius: 6px;
    border: 1px solid #38bdf8;
    background: #0ea5e9;
    color: #04121f;
    font: inherit;
    font-weight: 700;
    cursor: pointer;
  }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  #out {
    margin-top: 18px;
    height: 62vh;
    overflow-y: auto;
    padding: 12px;
    border: 1px solid #1e293b;
    border-radius: 8px;
    background: #0b1220;
    font-size: 13px;
    line-height: 1.55;
  }
  #out .row { display: flex; gap: 10px; align-items: baseline; padding: 1px 0; }
  #out .tag {
    flex: 0 0 auto;
    min-width: 76px;
    padding: 1px 7px;
    border-radius: 4px;
    color: #06121f;
    font-size: 11px;
    font-weight: 700;
    text-align: center;
  }
  #out .detail { white-space: pre-wrap; word-break: break-word; color: #cbd5e1; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <p class="sub">Autonomous coding loop &mdash; start a goal and watch the run stream live.</p>

  <form id="run-form" autocomplete="off">
    <input id="goal" name="goal" placeholder="Describe the goal for Furrow" required />
    <input id="workspace" name="workspace" class="ws" placeholder="workspace (optional)" />
    <button id="start" type="submit">Start</button>
  </form>

  <div id="out"></div>

<script>
(function () {
  var form = document.getElementById('run-form');
  var goalInput = document.getElementById('goal');
  var workspaceInput = document.getElementById('workspace');
  var startBtn = document.getElementById('start');
  var out = document.getElementById('out');

  var COLORS = {
    start: '#38bdf8', cycle: '#a78bfa', plan: '#60a5fa',
    task_start: '#fbbf24', task_done: '#34d399', edit: '#f472b6',
    test: '#22d3ee', done: '#4ade80', stopped: '#facc15',
    error: '#f87171', info: '#64748b'
  };

  function formatDetail(payload) {
    var copy = {};
    Object.keys(payload || {}).forEach(function (key) {
      if (key !== 'event') { copy[key] = payload[key]; }
    });
    var text;
    try {
      text = JSON.stringify(copy);
    } catch (err) {
      text = String(copy);
    }
    if (text === undefined) { text = ''; }
    if (text.length > 400) { text = text.slice(0, 400) + '...'; }
    return text;
  }

  function addLine(event, payload) {
    var row = document.createElement('div');
    row.className = 'row';

    var tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = event;
    tag.style.background = COLORS[event] || '#475569';

    var detail = document.createElement('span');
    detail.className = 'detail';
    detail.textContent = formatDetail(payload);

    row.appendChild(tag);
    row.appendChild(detail);
    out.appendChild(row);
    out.scrollTop = out.scrollHeight;
  }

  var socket = null;

  function setRunning(running) {
    startBtn.disabled = running;
    startBtn.textContent = running ? 'Running...' : 'Start';
  }

  form.addEventListener('submit', function (ev) {
    ev.preventDefault();
    if (socket) { return; }

    var goal = goalInput.value.trim();
    if (!goal) {
      goalInput.focus();
      return;
    }

    var proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    var ws = new WebSocket(proto + '://' + window.location.host + '/ws');
    socket = ws;
    setRunning(true);
    addLine('info', { status: 'connecting' });

    ws.onopen = function () {
      var workspace = workspaceInput.value.trim();
      ws.send(JSON.stringify({ goal: goal, workspace: workspace || null }));
    };

    ws.onmessage = function (message) {
      var payload;
      try {
        payload = JSON.parse(message.data);
      } catch (err) {
        addLine('error', { error: String(message.data) });
        return;
      }
      if (payload && payload.event) {
        addLine(String(payload.event), payload);
      } else {
        addLine('info', payload);
      }
    };

    ws.onerror = function () {
      addLine('error', { error: 'websocket error' });
    };

    ws.onclose = function () {
      addLine('info', { status: 'disconnected' });
      socket = null;
      setRunning(false);
    };
  });
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _json_safe(payload: Any) -> Any:
    try:
        return json.loads(json.dumps(payload, default=str))
    except (TypeError, ValueError):
        return str(payload)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    receiver: asyncio.Task | None = None
    runner: asyncio.Task | None = None
    inflight: set[asyncio.Task] = set()

    async def send(message: dict) -> None:
        try:
            await websocket.send_json(message)
        except Exception:
            pass

    def emit(event: str, payload: dict) -> None:
        """Forward an orchestrator event to the client (called from async code)."""
        body = _json_safe(payload if isinstance(payload, dict) else {"data": payload})
        if not isinstance(body, dict):
            body = {"data": body}
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(send({"event": event, **body}))
        inflight.add(task)
        task.add_done_callback(inflight.discard)

    async def listen() -> None:
        """Consume client traffic so disconnects surface promptly."""
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    try:
        data = await websocket.receive_json()
        goal = str(data.get("goal") or "").strip()
        if not goal:
            await websocket.send_json({"event": "error", "error": "goal is required"})
            await websocket.close()
            return

        workspace = data.get("workspace")
        run_settings = (
            Settings(workspace=Path(str(workspace))) if workspace else default_settings
        )

        receiver = asyncio.create_task(listen())
        orchestrator = Orchestrator(goal=goal, settings=run_settings, on_event=emit)
        runner = asyncio.create_task(orchestrator.run())

        done, pending = await asyncio.wait(
            {receiver, runner}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(BaseException):
                await task
        if runner in done:
            await runner
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"event": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        if receiver is not None and not receiver.done():
            receiver.cancel()
            with contextlib.suppress(BaseException):
                await receiver
        if runner is not None and not runner.done():
            runner.cancel()
            with contextlib.suppress(BaseException):
                await runner
        if inflight:
            with contextlib.suppress(BaseException):
                await asyncio.wait(list(inflight), timeout=1.0)
        with contextlib.suppress(Exception):
            await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)