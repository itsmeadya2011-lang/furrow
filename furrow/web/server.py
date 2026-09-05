from __future__ import annotations

import json
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Furrow</title>
  <style>
    :root {
      --bg: #0f1115;
      --panel: #161a22;
      --border: #2a2f3a;
      --text: #e6e8ee;
      --muted: #8b93a7;
      --accent: #4f8cff;
      --accent-hover: #3a6fd8;
      --success: #2dd4a8;
      --error: #f25f5c;
      --warn: #facc15;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      height: 100%;
    }
    .container {
      max-width: 960px;
      margin: 0 auto;
      padding: 24px 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      min-height: 100vh;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      letter-spacing: 0.5px;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--panel);
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 12px;
    }
    .status .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--muted);
    }
    .status.running { color: var(--warn); }
    .status.running .dot { background: var(--warn); }
    .status.complete { color: var(--success); }
    .status.complete .dot { background: var(--success); }
    .status.error { color: var(--error); }
    .status.error .dot { background: var(--error); }
    form {
      display: flex;
      gap: 10px;
    }
    input[type="text"] {
      flex: 1 1 auto;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      outline: none;
    }
    input[type="text"]:focus { border-color: var(--accent); }
    button {
      padding: 10px 14px;
      border-radius: 8px;
      border: 1px solid transparent;
      background: var(--accent);
      color: white;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { background: var(--accent-hover); }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .log-wrap {
      flex: 1 1 auto;
      min-height: 0;
      display: flex;
      flex-direction: column;
    }
    .log {
      flex: 1 1 auto;
      min-height: 320px;
      max-height: 70vh;
      overflow: auto;
      padding: 12px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .log .event { color: var(--text); }
    .log .status { color: var(--muted); }
    .log .error { color: var(--error); }
    .log .done { color: var(--success); }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Furrow</h1>
      <div id="status" class="status">
        <span id="statusDot" class="dot"></span>
        <span id="statusText">idle</span>
      </div>
    </header>
    <form id="form">
      <input id="goal" placeholder="Enter goal" autocomplete="off" required />
      <button id="startBtn" type="submit">Start</button>
    </form>
    <div class="log-wrap">
      <pre id="out" class="log"></pre>
    </div>
  </div>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const startBtn = document.getElementById('startBtn');
    const status = document.getElementById('status');
    const statusText = document.getElementById('statusText');

    function setStatus(state, text) {
      status.className = 'status ' + state;
      statusText.textContent = text;
    }

    function append(cls, text) {
      const node = document.createElement('div');
      node.className = cls;
      node.textContent = text;
      out.appendChild(node);
      out.scrollTop = out.scrollHeight;
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      startBtn.disabled = true;
      out.innerHTML = '';
      setStatus('running', 'Running...');
      const goal = document.getElementById('goal').value;
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => ws.send(JSON.stringify({goal}));
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'event') {
            append('event', msg.data);
          } else if (msg.type === 'status') {
            setStatus('running', msg.data || 'Running...');
          } else if (msg.type === 'error') {
            append('error', msg.data || 'Error');
            setStatus('error', 'Error');
          } else if (msg.type === 'done') {
            setStatus('complete', 'Complete');
            append('done', 'Done');
          }
        } catch (_) {
          append('event', ev.data);
        }
      };
      ws.onclose = () => {
        if (!status.classList.contains('complete') && !status.classList.contains('error')) {
          setStatus('', 'Done');
        }
        startBtn.disabled = false;
      };
      ws.onerror = () => {
        append('error', 'Connection error');
        setStatus('error', 'Error');
        startBtn.disabled = false;
      };
    };
  </script>
</body>
</html>""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        async def send_json(message: dict[str, Any]) -> None:
            try:
                await websocket.send_json(message)
            except Exception:
                pass

        def on_event(message: str) -> None:
            send_json({"type": "event", "data": message})

        client_settings = Settings()
        if model:
            client_settings.model = model
            client_settings.planner_model = model
            client_settings.worker_model = model
            client_settings.tester_model = model
        client = LLMClient(settings=client_settings)
        orchestrator = Orchestrator(goal=goal, on_event=on_event, client=client)

        try:
            await orchestrator.run()
            await send_json({"type": "done"})
        except Exception as exc:
            await send_json({"type": "error", "data": f"{exc}"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "data": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
