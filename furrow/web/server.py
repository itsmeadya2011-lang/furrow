from __future__ import annotations

import asyncio
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
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
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
    queue: asyncio.Queue[str] = asyncio.Queue()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        async def on_output(line: str) -> None:
            await queue.put(line)

        orchestrator = Orchestrator(goal=goal, on_output=on_output)
        loop_task = asyncio.create_task(orchestrator.run())

        while True:
            if loop_task.done() and queue.empty():
                break
            try:
                line = await asyncio.wait_for(queue.get(), timeout=0.5)
                await websocket.send_text(line + "\n")
            except asyncio.TimeoutError:
                continue

        while not queue.empty():
            line = await queue.get()
            await websocket.send_text(line + "\n")

        await websocket.send_text("\nDone.\n")
    except WebSocketDisconnect:
        pass
    finally:
        if "loop_task" in locals() and not loop_task.done():
            loop_task.cancel()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
