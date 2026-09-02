"""Furrow web server exposing a browser UI over FastAPI + WebSocket."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None
    resume: bool = False


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<title>Furrow</title>
<style>
  body { font-family: sans-serif; margin: 24px; }
  #form { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }
  input[type=text] { flex: 1; padding: 8px; font-size: 14px; }
  button { padding: 8px 14px; font-size: 14px; }
  #status { margin-bottom: 10px; padding: 8px; border: 1px solid #ccc; }
  #log { width: 100%; height: 420px; border: 1px solid #ccc; padding: 8px; overflow: auto; background: #0e1116; color: #c9d1d9; font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
  .ev-cycle_start { color: #58a6ff; }
  .ev-plan_ready { color: #bc8cff; }
  .ev-task_start { color: #d29922; }
  .ev-task_complete { color: #3fb950; }
  .ev-task_failed { color: #f85149; }
  .ev-test_complete { color: #79c0ff; }
  .ev-cycle_end { color: #8b949e; }
  .ev-done { color: #3fb950; }
  .ev-_end { color: #3fb950; }
  .ev-_error { color: #f85149; }
  .ev-default { color: #c9d1d9; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <input id="model" placeholder="Model (optional)" />
    <label><input id="resume" type="checkbox" /> Resume</label>
    <button type="submit">Start</button>
  </form>
  <div id="status">Idle</div>
  <pre id="log"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('log');
    const status = document.getElementById('status');
    let eventCount = 0;
    let cycle = 0;
    let doneReason = '';

    function append(type, text) {
      const line = document.createElement('div');
      line.className = 'ev-' + type.replace(/[^a-zA-Z0-9_-]/g, '_');
      line.textContent = text;
      out.appendChild(line);
      out.scrollTop = out.scrollHeight;
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      eventCount = 0;
      cycle = 0;
      doneReason = '';
      status.textContent = 'Running...';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          const type = msg.type || 'default';
          const data = msg.data || {};
          eventCount += 1;
          if (data.cycle) cycle = data.cycle;
          if (data.reason) doneReason = data.reason;
          if (type === '_end' || type === 'done') {
            status.textContent = 'Done. Cycle: ' + cycle + ', Reason: ' + doneReason + ', Events: ' + eventCount;
          } else if (type === '_error') {
            status.textContent = 'Error: ' + (data.message || 'unknown');
          } else {
            status.textContent = 'Cycle: ' + cycle + ', Events: ' + eventCount;
          }
          append(type, '[' + type + '] ' + JSON.stringify(data));
        } catch (err) {
          append('default', ev.data);
        }
      };
      ws.onclose = () => { status.textContent = 'Closed. Cycle: ' + cycle + ', Events: ' + eventCount; };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value, model: document.getElementById('model').value || undefined, resume: document.getElementById('resume').checked}));
    };
  </script>
</body>
</html>"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        # TODO: wire `resume` to StateStore-based resumption once the
        # orchestrator supports loading a FurrowState on init.
        _resume = bool(data.get("resume", False))

        # Apply model override to the shared settings singleton so the
        # LLMClient created by the orchestrator picks it up.
        if model:
            from furrow.config import settings as shared_settings
            shared_settings.model = model

        stop_event = asyncio.Event()

        async def on_event(ev: str, payload: dict) -> None:
            try:
                coro = websocket.send_json({"type": ev, "data": payload})
                if asyncio.iscoroutine(coro):
                    await coro
            except Exception:
                pass

        orchestrator = Orchestrator(
            goal=goal,
            on_event=on_event,
            stop_event=stop_event,
        )
        try:
            await orchestrator.run()
        except Exception as e:
            try:
                await websocket.send_json({"type": "_error", "data": {"message": str(e)}})
            except Exception:
                pass
        finally:
            reason = getattr(orchestrator, "_done_reason", None) or "complete"
            try:
                await websocket.send_json({"type": "_end", "data": {"reason": reason}})
            except Exception:
                pass
            try:
                await websocket.close()
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
