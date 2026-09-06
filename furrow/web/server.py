from __future__ import annotations

import asyncio
import contextlib
import io
import json
import traceback
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Furrow</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      max-width: 900px;
      margin: 0 auto;
      padding: 2rem;
      min-height: 100vh;
    }
    h1 { color: #38bdf8; margin-bottom: 0.5rem; }
    .subtitle { color: #94a3b8; margin-bottom: 1.5rem; }
    .status-bar {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 1.5rem;
      padding: 0.75rem 1rem;
      background: #1e293b;
      border-radius: 0.5rem;
      border: 1px solid #334155;
    }
    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #64748b;
    }
    .status-dot.connected { background: #22c55e; }
    .status-dot.error { background: #ef4444; }
    .status-dot.running { background: #eab308; }
    .form-group {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1.5rem;
    }
    input[type="text"] {
      flex: 1;
      padding: 0.75rem 1rem;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 0.5rem;
      color: #e2e8f0;
      font-size: 1rem;
    }
    input[type="text"]:focus { outline: none; border-color: #38bdf8; }
    button {
      padding: 0.75rem 1.5rem;
      background: #0ea5e9;
      color: white;
      border: none;
      border-radius: 0.5rem;
      cursor: pointer;
      font-size: 1rem;
      font-weight: 500;
    }
    button:hover { background: #0284c7; }
    button:disabled { background: #475569; cursor: not-allowed; }
    .disconnect-btn { background: #ef4444; margin-left: 0.5rem; }
    .disconnect-btn:hover { background: #dc2626; }
    .log-container {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 0.5rem;
      padding: 1rem;
      height: 500px;
      overflow-y: auto;
      font-family: 'SF Mono', Menlo, monospace;
      font-size: 0.875rem;
      line-height: 1.5;
    }
    .log-entry { margin-bottom: 0.25rem; }
    .log-entry .timestamp { color: #64748b; margin-right: 0.5rem; }
    .log-entry.info { color: #38bdf8; }
    .log-entry.success { color: #22c55e; }
    .log-entry.error { color: #ef4444; }
    .log-entry.warning { color: #eab308; }
    .log-entry.system { color: #94a3b8; font-style: italic; }
    .empty-state { color: #64748b; text-align: center; padding: 3rem; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <p class="subtitle">AI-powered autonomous coding agent</p>
  <div class="status-bar">
    <div class="status-dot" id="statusDot"></div>
    <span id="statusText">Disconnected</span>
    <button id="disconnectBtn" class="disconnect-btn" style="display:none;">Disconnect</button>
  </div>
  <form id="form" class="form-group">
    <input id="goal" placeholder="Enter goal (e.g., create a REST API)" required />
    <button type="submit" id="startBtn">Start</button>
  </form>
  <div class="log-container" id="out">
    <div class="empty-state">No messages yet. Enter a goal and click Start.</div>
  </div>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const startBtn = document.getElementById('startBtn');
    const disconnectBtn = document.getElementById('disconnectBtn');
    let ws = null;

    function setStatus(state, text) {
      statusDot.className = 'status-dot ' + state;
      statusText.textContent = text;
    }

    function appendMessage(data, type) {
      const div = document.createElement('div');
      div.className = 'log-entry ' + (type || 'info');
      const ts = new Date().toLocaleTimeString();
      div.innerHTML = '<span class="timestamp">[' + ts + ']</span>' + escapeHtml(data);
      out.appendChild(div);
      out.scrollTop = out.scrollHeight;
      const empty = out.querySelector('.empty-state');
      if (empty) empty.remove();
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    function disconnect() {
      if (ws) {
        ws.close();
        ws = null;
      }
      setStatus('', 'Disconnected');
      startBtn.disabled = false;
      disconnectBtn.style.display = 'none';
    }

    disconnectBtn.onclick = disconnect;

    form.onsubmit = async (e) => {
      e.preventDefault();
      disconnect();
      appendMessage('Starting connection...', 'system');
      const wsUrl = 'ws://' + location.host + '/ws';
      ws = new WebSocket(wsUrl);
      ws.onopen = () => {
        setStatus('connected', 'Connected');
        startBtn.disabled = true;
        disconnectBtn.style.display = 'inline-block';
        appendMessage('Sending goal: ' + document.getElementById('goal').value, 'system');
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'status') {
            appendMessage(msg.message, 'info');
          } else if (msg.type === 'error') {
            appendMessage(msg.message, 'error');
          } else {
            appendMessage(ev.data, 'info');
          }
        } catch {
          appendMessage(ev.data, 'info');
        }
      };
      ws.onerror = () => {
        appendMessage('Connection error', 'error');
        setStatus('error', 'Connection error');
      };
      ws.onclose = () => {
        appendMessage('Connection closed', 'system');
        setStatus('', 'Disconnected');
        startBtn.disabled = false;
        disconnectBtn.style.display = 'none';
      };
    };
  </script>
</body>
</html>
""")


@app.post("/start")
async def start_endpoint(request: StartRequest) -> dict:
    result = {"status": "accepted", "goal": request.goal, "model": request.model or "default"}
    return result


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    is_running = False

    async def send_json(data: dict) -> None:
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    async def send_text(text: str) -> None:
        try:
            await websocket.send_text(text)
        except Exception:
            pass

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        if is_running:
            await send_json({"type": "error", "message": "An orchestrator is already running."})
            return

        is_running = True
        await send_json({"type": "status", "message": f"Starting Furrow... Goal: {goal}"})

        orchestrator = Orchestrator(goal=goal)
        stdout_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture):
                await orchestrator.run()
            await send_json({"type": "status", "message": "Furrow completed."})
        except Exception as exc:
            await send_json({
                "type": "error",
                "message": f"Error: {exc}",
                "traceback": traceback.format_exc(),
            })
        finally:
            captured = stdout_capture.getvalue()
            if captured:
                for line in captured.splitlines():
                    if line.strip():
                        await send_text(line)
            is_running = False

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await send_json({"type": "error", "message": f"Connection error: {exc}"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
