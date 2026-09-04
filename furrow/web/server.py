from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
    h1 { color: #1a1a1a; }
    form { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
    input { flex: 1; padding: 0.5rem; font-size: 1rem; border: 1px solid #ccc; border-radius: 4px; }
    button { padding: 0.5rem 1rem; font-size: 1rem; background: #0d6; color: white; border: none; border-radius: 4px; cursor: pointer; }
    button:disabled { background: #999; }
    #out { background: #111; color: #0f0; padding: 1rem; border-radius: 4px; font-family: monospace; white-space: pre-wrap; min-height: 200px; }
    .error { color: #f55; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter your goal" required />
    <button type="submit" id="btn">Start</button>
  </form>
  <pre id="out">Ready. Enter a goal and click Start.</pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const btn = document.getElementById('btn');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      btn.disabled = true;
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      ws.onmessage = (ev) => {
        const msg = ev.data;
        out.textContent += msg + '\\n';
        out.scrollTop = out.scrollHeight;
      };
      ws.onerror = () => { out.innerHTML += '<span class="error">Connection error</span>\\n'; btn.disabled = false; };
      ws.onclose = () => { btn.disabled = false; };
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)

        async def on_progress(message: str) -> None:
            await websocket.send_text(message)

        orchestrator.on_progress = on_progress
        await orchestrator.run()
        await websocket.send_text("Done")
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(f"Error: {exc}")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
