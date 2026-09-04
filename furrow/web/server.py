from __future__ import annotations

import asyncio
import io
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Furrow</title>
  <style>
    :root {
      --bg: #0f1115;
      --panel: #161a22;
      --border: #2a3140;
      --fg: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --green: #3fb950;
      --red: #f85149;
      --yellow: #d29922;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--fg);
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
      display: flex;
      align-items: center;
      gap: 16px;
    }
    header h1 {
      margin: 0;
      font-size: 20px;
      letter-spacing: 0.5px;
    }
    header .badge {
      font-size: 12px;
      padding: 2px 8px;
      border-radius: 10px;
      background: var(--border);
      color: var(--muted);
    }
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 16px 24px;
      gap: 12px;
      overflow: hidden;
    }
    form {
      display: flex;
      gap: 8px;
    }
    input[type=text] {
      flex: 1;
      padding: 10px 12px;
      background: var(--panel);
      color: var(--fg);
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 14px;
      outline: none;
    }
    input[type=text]:focus { border-color: var(--accent); }
    button {
      padding: 10px 18px;
      background: var(--accent);
      color: #fff;
      border: 0;
      border-radius: 6px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
    }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .progress {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
      min-height: 22px;
    }
    .spinner {
      width: 14px;
      height: 14px;
      border: 2px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      display: none;
    }
    .spinner.active { display: inline-block; }
    @keyframes spin { to { transform: rotate(360deg); } }
    #out {
      flex: 1;
      margin: 0;
      padding: 12px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow-y: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .event { padding: 4px 0; border-bottom: 1px dashed var(--border); }
    .event:last-child { border-bottom: 0; }
    .tag {
      display: inline-block;
      padding: 1px 6px;
      border-radius: 4px;
      font-size: 11px;
      margin-right: 6px;
      font-weight: 600;
    }
    .tag.plan { background: #1f6feb33; color: var(--accent); }
    .tag.started { background: #d2992233; color: var(--yellow); }
    .tag.completed { background: #3fb95033; color: var(--green); }
    .tag.failed { background: #f8514933; color: var(--red); }
    .tag.tests_passed { background: #3fb95033; color: var(--green); }
    .tag.tests_failed { background: #f8514933; color: var(--red); }
    .tag.goal_complete { background: #a371f733; color: #a371f7; }
    .tag.log { background: var(--border); color: var(--muted); }
    .detail { color: var(--muted); margin-left: 4px; }
  </style>
</head>
<body>
  <header>
    <h1>Furrow</h1>
    <span class="badge">streaming</span>
  </header>
  <main>
    <form id="form">
      <input id="goal" type="text" placeholder="Describe a goal..." required autocomplete="off" />
      <button id="start" type="submit">Start</button>
    </form>
    <div class="progress">
      <span class="spinner" id="spin"></span>
      <span id="status">Idle</span>
    </div>
    <pre id="out"></pre>
  </main>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const status = document.getElementById('status');
    const spin = document.getElementById('spin');
    const startBtn = document.getElementById('start');

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = proto + '://' + location.host + '/ws';

    function setStatus(text, busy) {
      status.textContent = text;
      spin.classList.toggle('active', !!busy);
    }

    function append(text) {
      out.textContent += text + '\\n';
      out.scrollTop = out.scrollHeight;
    }

    function appendEvent(type, data) {
      const div = document.createElement('div');
      div.className = 'event';
      const tag = document.createElement('span');
      tag.className = 'tag ' + type;
      tag.textContent = type;
      div.appendChild(tag);
      const detail = document.createElement('span');
      detail.className = 'detail';
      if (type === 'plan') {
        detail.textContent = (data.rationale || '') + ' — ' + (data.tasks || []).length + ' task(s)';
      } else if (type === 'task_started' || type === 'task_completed' || type === 'task_failed') {
        detail.textContent = '#' + (data.id || '?') + ' ' + (data.description || '');
        if (data.error) detail.textContent += ' — ' + data.error;
      } else if (type === 'tests_passed') {
        detail.textContent = data.summary || '';
      } else if (type === 'tests_failed') {
        detail.textContent = (data.summary || '') + ' (' + (data.failures || []).length + ' failure(s))';
      } else if (type === 'goal_complete') {
        detail.textContent = 'after ' + (data.cycles || 0) + ' cycle(s)';
      } else if (type === 'log') {
        detail.textContent = data.text || '';
      } else {
        detail.textContent = JSON.stringify(data);
      }
      div.appendChild(detail);
      out.appendChild(div);
      out.scrollTop = out.scrollHeight;
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value.trim();
      if (!goal) return;
      out.textContent = '';
      startBtn.disabled = true;
      setStatus('Connecting...', true);

      const ws = new WebSocket(wsUrl);
      ws.onopen = () => {
        setStatus('Running...', true);
        ws.send(JSON.stringify({ goal }));
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          appendEvent(msg.type, msg.data || {});
          if (msg.type === 'goal_complete') setStatus('Done', false);
          else if (msg.type === 'task_failed' || msg.type === 'tests_failed') setStatus('Running (with errors)...', true);
        } catch {
          append(ev.data);
        }
      };
      ws.onerror = () => setStatus('Connection error', false);
      ws.onclose = () => {
        setStatus('Closed', false);
        startBtn.disabled = false;
      };
    };
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    log_buffer = io.StringIO()

    async def send(event_type: str, data: dict) -> None:
        try:
            await websocket.send_json({"type": event_type, "data": data})
        except Exception:
            pass

    def on_event(event_type: str, data: dict) -> None:
        coro = send(event_type, data)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
        else:
            loop.create_task(coro)

    def on_log(text: str) -> None:
        if not text:
            return
        coro = send("log", {"text": text})
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
        else:
            loop.create_task(coro)

    stream_console = Console(
        file=log_buffer,
        force_terminal=False,
        width=120,
        soft_wrap=True,
    )

    original_emit = stream_console.print

    def stream_print(*args, **kwargs):
        original_emit(*args, **kwargs)
        text = log_buffer.getvalue()
        if text:
            log_buffer.seek(0)
            log_buffer.truncate(0)
            on_log(text)

    stream_console.print = stream_print  # type: ignore[method-assign]

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(
            goal=goal,
            on_event=on_event,
            console=stream_console,
        )
        await orchestrator.run()
        try:
            await websocket.close()
        except Exception:
            pass
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "data": {"message": str(exc)}})
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)