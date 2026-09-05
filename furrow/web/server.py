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
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          const line = document.createElement('div');
          line.style.color = msg.color || 'inherit';
          line.textContent = '[' + msg.type + '] ' + msg.text;
          out.appendChild(line);
        } catch {
          out.textContent += ev.data + '\\n';
        }
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[dict] = asyncio.Queue()

    def on_output(text: str) -> None:
        type_ = "log"
        color = "inherit"
        lower = text.lower()
        if "goal complete" in lower or "tests passed" in lower:
            type_ = "success"
            color = "green"
        elif "tests failed" in lower or "failed" in lower:
            type_ = "error"
            color = "red"
        elif "plan" in lower:
            type_ = "plan"
            color = "blue"
        elif "cycle" in lower:
            type_ = "cycle"
            color = "cyan"
        asyncio.create_task(queue.put({"type": type_, "text": text, "color": color}))

    sender_task = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal, on_output=on_output)

        async def sender() -> None:
            while True:
                msg = await queue.get()
                await websocket.send_json(msg)

        sender_task = asyncio.create_task(sender())
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    finally:
        if sender_task is not None:
            sender_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
