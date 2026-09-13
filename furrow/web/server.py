from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# In-memory registry of active sessions for status monitoring
_active_sessions: dict[str, dict[str, Any]] = {}


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <p>WebSocket endpoint: <code>/ws</code></p>
  <p>Status: <code>/api/status</code></p>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        out.textContent += JSON.stringify(data) + '\\n';
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


@app.get("/api/status")
async def status() -> JSONResponse:
    return JSONResponse({"sessions": _active_sessions})


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(id(websocket))
    _active_sessions[session_id] = {"status": "starting", "goal": "", "cycles": 0}

    async def on_event(event: str, **data: Any) -> None:
        payload = {"event": event, **data}
        try:
            await websocket.send_text(json.dumps(payload, default=str))
        except Exception:
            pass
        # Update session state
        if event == "start":
            _active_sessions[session_id]["goal"] = data.get("goal", "")
        elif event == "cycle_start":
            _active_sessions[session_id]["cycles"] = data.get("cycle", 0)
        elif event in ("done", "plan_error"):
            _active_sessions[session_id]["status"] = event
        elif event == "cycle_end":
            _active_sessions[session_id]["cycles"] = data.get("cycle", 0)

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        session_settings = Settings()
        if model:
            session_settings.model = model

        orchestrator = Orchestrator(
            goal=goal,
            settings=session_settings,
            on_event=on_event,
        )
        _active_sessions[session_id]["status"] = "running"
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(
                json.dumps({"event": "error", "message": str(e)})
            )
        except Exception:
            pass
    finally:
        _active_sessions.pop(session_id, None)
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
