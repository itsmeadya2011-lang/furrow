from __future__ import annotations

import asyncio
import json
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


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Furrow</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f1115;
      color: #e6e6e6;
      margin: 0;
      padding: 2rem;
      max-width: 900px;
      margin-left: auto;
      margin-right: auto;
    }
    h1 { color: #6ee7b7; margin-top: 0; }
    form { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
    input[type="text"] {
      flex: 1;
      padding: 0.5rem;
      font-size: 1rem;
      border: 1px solid #3a3f47;
      border-radius: 4px;
      background: #1b1e23;
      color: #e6e6e6;
    }
    button {
      padding: 0.5rem 1rem;
      font-size: 1rem;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      background: #6ee7b7;
      color: #0f1115;
      font-weight: 600;
    }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    button.stop { background: #f87171; color: #0f1115; }
    #status {
      margin: 0.5rem 0;
      padding: 0.5rem;
      background: #1b1e23;
      border: 1px solid #3a3f47;
      border-radius: 4px;
      min-height: 2.5rem;
      font-family: monospace;
      white-space: pre-wrap;
    }
    pre#out {
      background: #1b1e23;
      border: 1px solid #3a3f47;
      border-radius: 4px;
      padding: 1rem;
      max-height: 60vh;
      overflow: auto;
      font-family: monospace;
      white-space: pre-wrap;
      margin: 0;
    }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <div id="status"></div>
  <button id="stop" class="stop" disabled>Stop</button>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const status = document.getElementById('status');
    const stopBtn = document.getElementById('stop');
    let ws = null;

    function setStatus(msg) {
      status.textContent = msg;
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      setStatus('Starting...');
      stopBtn.disabled = false;
      const goal = document.getElementById('goal').value;
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => {
        ws.send(JSON.stringify({ goal }));
      };
      ws.onmessage = (ev) => {
        let data;
        try { data = JSON.parse(ev.data); } catch { data = { message: ev.data }; }
        if (data.type === 'status') {
          setStatus(data.message);
        } else if (data.type === 'log') {
          out.textContent += data.message + '\\n';
        } else {
          out.textContent += ev.data + '\\n';
        }
      };
      ws.onclose = () => {
        setStatus('Closed.');
        stopBtn.disabled = true;
      };
      ws.onerror = (err) => {
        out.textContent += '\\n[error] WebSocket error\\n';
      };
    };

    stopBtn.onclick = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'stop' }));
        setStatus('Stopping...');
      }
    };
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    stop_event = asyncio.Event()
    orchestrator: Orchestrator | None = None
    task: asyncio.Task | None = None

    async def send_status(message: str) -> None:
        try:
            await websocket.send_json({"type": "status", "message": message})
        except Exception:
            pass

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal, stop_event=stop_event, on_status=send_status)

        async def run_and_report() -> None:
            assert orchestrator is not None
            try:
                await orchestrator.run()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await send_status(f"Error: {exc}")

        task = asyncio.create_task(run_and_report())

        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                break

            if isinstance(msg, dict) and msg.get("action") == "stop":
                stop_event.set()
                break

        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if task is not None and task.done() and not task.cancelled():
            try:
                task.result()
            except Exception:
                pass

        await send_status("Done.")
    except WebSocketDisconnect:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "log", "message": f"Error: {exc}"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)