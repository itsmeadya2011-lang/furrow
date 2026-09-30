from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Furrow</title>
  <style>
    body { font-family: sans-serif; margin: 1rem; }
    #out {
      background: #111;
      color: #eee;
      padding: 0.5rem;
      border-radius: 4px;
      max-height: 70vh;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required style="width:70%" />
    <input id="model" placeholder="Model (optional)" size="30" />
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
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      const payload = { goal: document.getElementById('goal').value };
      const m = document.getElementById('model').value.trim();
      if (m) { payload.model = m; }
      ws.send(JSON.stringify(payload));
    };
  </script>
</body>
</html>
"""


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data: Any = await websocket.receive_json()
        goal = str(data.get("goal", ""))
        model = data.get("model")
        if model:
            settings.model = str(model)

        async def broadcaster(msg: str) -> None:
            await websocket.send_text(msg)

        orchestrator = Orchestrator(goal=goal, reporter=broadcaster)
        await orchestrator.run()
        await websocket.send_text("[Furrow] Done.")
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        await websocket.send_text(f"[Furrow] Error: {exc}")
    finally:
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
