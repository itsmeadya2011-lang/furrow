from __future__ import annotations

import asyncio
import re
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


RICH_TAG_RE = re.compile(r"\[/?[a-zA-Z0-9 #]+\]")


def strip_rich_markup(text: str) -> str:
    return RICH_TAG_RE.sub("", text)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    let ws = null;
    let reconnecting = false;

    function connect() {
      if (ws) return;
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        if (ev.data === '__DONE__') {
          out.textContent += '\\n--- Done ---\\n';
          ws.close();
          ws = null;
          return;
        }
        out.textContent += ev.data + '\\n';
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => {
        if (!reconnecting) {
          out.textContent += '\\nClosed.\\n';
        }
        ws = null;
      };
      ws.onerror = () => {
        out.textContent += '\\nConnection error.\\n';
      };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      connect();
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

        queue: asyncio.Queue[str] = asyncio.Queue()

        def on_output(message: str, renderable: Optional[object] = None) -> None:
            clean = strip_rich_markup(message)
            try:
                queue.put_nowait(clean)
            except asyncio.QueueFull:
                pass

        orchestrator = Orchestrator(goal=goal, on_output=on_output)
        run_task = asyncio.create_task(orchestrator.run())

        while not run_task.done():
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=0.1)
                await websocket.send_text(msg)
            except asyncio.TimeoutError:
                continue

        await run_task
        await websocket.send_text("__DONE__")
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
