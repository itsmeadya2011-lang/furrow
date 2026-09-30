from __future__ import annotations

import json
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

# Configure structured logging to stdout for the web server
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: monospace; margin: 2em; }
    #out { background: #1e1e1e; color: #d4d4d4; padding: 1em; border-radius: 6px;
           height: 60vh; overflow-y: auto; white-space: pre-wrap; }
    #form { margin-bottom: 1em; }
    input, button { padding: 0.5em; font-size: 1em; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required style="width: 600px;" />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          out.textContent += JSON.stringify(msg, null, 2) + '\\n';
        } catch {
          out.textContent += ev.data + '\\n';
        }
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        settings = Settings()
        if model:
            settings.model = model

        connected = True

        async def reporter(event: str, payload: dict[str, Any]) -> None:
            if not connected:
                return
            await websocket.send_text(json.dumps({"event": event, **payload}, default=str))

        orchestrator = Orchestrator(goal=goal, settings=settings)
        await orchestrator.run(reporter=reporter)
        await websocket.send_text(json.dumps({"event": "done", "goal": goal}))
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001 — send any error to the client
        await websocket.send_text(json.dumps({"event": "error", "error": str(e)}))
    finally:
        connected = False


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
