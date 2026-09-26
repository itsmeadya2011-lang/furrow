from __future__ import annotations

import json

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/")
async def index() -> HTMLResponse:
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Furrow</title>
  <style>
    body { font-family: ui-monospace, monospace; margin: 2rem; background: #0a0a0a; color: #e0e0e0; }
    h1 { color: #4CAF50; }
    input, button { padding: 0.5rem; font-size: 1rem; }
    input { width: 60%; background: #1a1a1a; color: #e0e0e0; border: 1px solid #444; }
    button { background: #4CAF50; color: #fff; border: none; cursor: pointer; }
    button:hover { background: #45a049; }
    pre { background: #1a1a1a; border: 1px solid #333; padding: 1rem; border-radius: 4px;
           max-height: 80vh; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }
    .event { margin: 2px 0; }
    .cycle { color: #2196F3; }
    .success { color: #4CAF50; }
    .error { color: #f44336; }
    .info { color: #9E9E9E; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal..." required />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');

    function formatEvent(data) {
      const el = document.createElement('div');
      el.className = 'event ' + (data.type || 'info');
      el.textContent = '[' + (data.type || 'info') + '] ' + JSON.stringify(data.payload || data);
      return el;
    }

    function append(text) {
      out.appendChild(document.createTextNode(text + '\\n'));
      out.scrollTop = out.scrollHeight;
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      out.innerHTML = '';
      append('Starting Furrow...');
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          out.appendChild(formatEvent(data));
        } catch {
          append(ev.data);
        }
        out.scrollTop = out.scrollHeight;
      };
      ws.onerror = (ev) => append('WebSocket error: ' + ev.message);
      ws.onclose = () => append('\\nClosed.\\n');
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        await websocket.send_json({"type": "error", "payload": {"error": "Invalid JSON received"}})
        await websocket.close()
        return

    goal = data.get("goal", "")
    model = data.get("model")

    if not goal:
        await websocket.send_json({"type": "error", "payload": {"error": "Missing 'goal' field"}})
        await websocket.close()
        return

    await websocket.send_json({"type": "startup", "payload": {"goal": goal}})

    settings = Settings()
    if model:
        settings.model = model

    async def emit(event_type: str, payload: object) -> None:
        await websocket.send_json({"type": event_type, "payload": payload})

    try:
        orchestrator = Orchestrator(goal=goal, settings=settings, on_event=emit)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        await websocket.send_json({"type": "error", "payload": {"error": str(exc)}})
    finally:
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
