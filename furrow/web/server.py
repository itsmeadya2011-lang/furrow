from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; }
  #out { background: #111; color: #0f0; padding: 1rem; white-space: pre-wrap; min-height: 200px; }
  button:disabled { opacity: 0.5; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required style="width: 70%" />
    <button id="start" type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const start = document.getElementById('start');
    form.onsubmit = async (e) => {
      e.preventDefault();
      start.disabled = true;
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        out.textContent += msg.data + '\\n';
        if (msg.type === 'done' || msg.type === 'error') {
          ws.close();
          start.disabled = false;
        }
      };
      ws.onclose = () => { start.disabled = false; };
      ws.onopen = () => {
        out.textContent += 'Starting...\\n';
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
    };
  </script>
</body>
</html>
""")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        async def send_event(message: str) -> None:
            await websocket.send_json({"type": "log", "data": message})

        orchestrator = Orchestrator(goal=goal, on_event=send_event)
        await orchestrator.run()
        await websocket.send_json({"type": "done", "data": "Furrow halted."})
    except WebSocketDisconnect:
        pass
    except Exception as e:  # pragma: no cover - surfaced to the client
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)