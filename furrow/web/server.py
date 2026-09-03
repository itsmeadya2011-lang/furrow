from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class SessionInfo(BaseModel):
    session_id: str
    goal: str
    status: str
    created_at: str
    updated_at: str
    cycles: int = 0
    error: Optional[str] = None


SESSIONS: dict[str, SessionInfo] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Furrow</title>
  <style>
    :root { color-scheme: light dark; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 900px;
      margin: 2rem auto;
      padding: 0 1rem;
      line-height: 1.5;
    }
    h1 { margin-bottom: 0.25rem; }
    .subtitle { color: #888; margin-top: 0; }
    #form { display: flex; gap: 0.5rem; margin: 1rem 0; }
    #goal {
      flex: 1;
      padding: 0.5rem 0.75rem;
      border: 1px solid #ccc;
      border-radius: 6px;
      font-size: 1rem;
    }
    button {
      padding: 0.5rem 1rem;
      border: none;
      border-radius: 6px;
      background: #2563eb;
      color: #fff;
      font-size: 1rem;
      cursor: pointer;
    }
    button.secondary { background: #6b7280; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .toolbar { display: flex; gap: 0.5rem; align-items: center; margin: 0.75rem 0; }
    .status {
      padding: 0.25rem 0.5rem;
      border-radius: 999px;
      font-size: 0.85rem;
      background: #e5e7eb;
      color: #111;
    }
    .status.running { background: #fde68a; color: #92400e; }
    .status.completed { background: #bbf7d0; color: #14532d; }
    .status.failed { background: #fecaca; color: #7f1d1d; }
    #out {
      background: #0b1020;
      color: #e6edf3;
      padding: 1rem;
      border-radius: 8px;
      min-height: 240px;
      max-height: 60vh;
      overflow: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.9rem;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .msg { display: block; padding: 2px 0; }
    .msg.info { color: #93c5fd; }
    .msg.success { color: #86efac; }
    .msg.warn { color: #fcd34d; }
    .msg.error { color: #fca5a5; font-weight: 600; }
    .msg.system { color: #c4b5fd; font-style: italic; }
    .ts { color: #6b7280; margin-right: 0.5rem; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <p class="subtitle">Real-time orchestration log</p>

  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button id="start" type="submit">Start</button>
  </form>

  <div class="toolbar">
    <span id="status" class="status">idle</span>
    <span id="sid" class="status"></span>
    <button id="clear" class="secondary" type="button">Clear</button>
  </div>

  <pre id="out"></pre>

  <script>
    const form = document.getElementById('form');
    const startBtn = document.getElementById('start');
    const clearBtn = document.getElementById('clear');
    const out = document.getElementById('out');
    const statusEl = document.getElementById('status');
    const sidEl = document.getElementById('sid');

    let ws = null;

    function setStatus(text, cls) {
      statusEl.textContent = text;
      statusEl.className = 'status' + (cls ? ' ' + cls : '');
    }

    function timestamp() {
      const d = new Date();
      return d.toLocaleTimeString();
    }

    function append(text, cls) {
      const span = document.createElement('span');
      span.className = 'msg' + (cls ? ' ' + cls : '');
      const ts = document.createElement('span');
      ts.className = 'ts';
      ts.textContent = '[' + timestamp() + ']';
      span.appendChild(ts);
      span.appendChild(document.createTextNode(text + '\\n'));
      out.appendChild(span);
      out.scrollTop = out.scrollHeight;
    }

    clearBtn.onclick = () => { out.textContent = ''; };

    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      if (ws) { ws.close(); ws = null; }
      out.textContent = '';
      setStatus('running', 'running');
      sidEl.textContent = '';
      startBtn.disabled = true;
      append('Starting session...', 'system');

      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => {
        append('Connected.', 'system');
        ws.send(JSON.stringify({goal}));
      };
      ws.onmessage = (ev) => {
        let payload = ev.data;
        let cls = 'info';
        try {
          const obj = JSON.parse(ev.data);
          payload = obj.message ?? ev.data;
          if (obj.type === 'error') cls = 'error';
          else if (obj.type === 'progress') {
            if (obj.event === 'task_completed' || obj.event === 'tests_passed' || obj.event === 'done') cls = 'success';
            else if (obj.event === 'task_failed' || obj.event === 'tests_failed') cls = 'error';
            else if (obj.event === 'cycle_start' || obj.event === 'tasks_start') cls = 'warn';
          } else if (obj.type === 'session') {
            sidEl.textContent = obj.session_id.slice(0, 8);
            cls = 'system';
            payload = 'Session ' + obj.session_id;
          } else if (obj.type === 'complete') {
            cls = 'success';
            payload = obj.message ?? 'Complete';
          }
          if (obj.session_id && !sidEl.textContent) {
            sidEl.textContent = obj.session_id.slice(0, 8);
          }
        } catch (_) {}
        append(payload, cls);
      };
      ws.onerror = () => {
        append('WebSocket error.', 'error');
        setStatus('failed', 'failed');
        startBtn.disabled = false;
      };
      ws.onclose = (ev) => {
        append('Closed (code ' + ev.code + ').', 'system');
        if (statusEl.textContent === 'running') {
          setStatus('disconnected');
        }
        startBtn.disabled = false;
      };
    };
  </script>
</body>
</html>
""")


@app.get("/sessions", response_model=list[SessionInfo])
async def list_sessions() -> list[SessionInfo]:
    return list(SESSIONS.values())


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str) -> SessionInfo:
    info = SESSIONS.get(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="session not found")
    return info


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = uuid4().hex
    goal = ""
    session = SessionInfo(
        session_id=session_id,
        goal="",
        status="running",
        created_at=_now(),
        updated_at=_now(),
    )
    SESSIONS[session_id] = session
    await websocket.send_json({"type": "session", "session_id": session_id, "message": f"Session {session_id} created"})

    async def progress(payload: dict[str, Any]) -> None:
        if payload.get("type") == "progress":
            session.cycles = max(session.cycles, int(payload.get("cycle", session.cycles) or 0))
            session.updated_at = _now()
        await websocket.send_json(payload)

    try:
        try:
            data = await websocket.receive_json()
        except WebSocketDisconnect:
            session.status = "disconnected"
            session.updated_at = _now()
            return

        goal = str(data.get("goal", ""))
        session.goal = goal
        session.updated_at = _now()
        await websocket.send_json({"type": "progress", "event": "received", "message": f"Goal: {goal}"})

        orchestrator = Orchestrator(goal=goal, progress_callback=progress)
        try:
            await orchestrator.run()
            session.status = "completed"
            session.updated_at = _now()
            await websocket.send_json({"type": "complete", "session_id": session_id, "message": "Run finished"})
        except Exception as exc:
            session.status = "failed"
            session.error = repr(exc)
            session.updated_at = _now()
            try:
                await websocket.send_json({
                    "type": "error",
                    "session_id": session_id,
                    "message": f"Orchestrator error: {exc!r}",
                })
            except Exception:
                pass
            try:
                await websocket.close(code=1011)
            except Exception:
                pass
            return

        try:
            await websocket.close(code=1000)
        except Exception:
            pass
    except WebSocketDisconnect:
        session.status = "disconnected"
        session.updated_at = _now()
    except Exception as exc:
        session.status = "failed"
        session.error = repr(exc)
        session.updated_at = _now()
        try:
            await websocket.send_json({
                "type": "error",
                "session_id": session_id,
                "message": f"Server error: {exc!r}",
            })
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        session.updated_at = _now()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)