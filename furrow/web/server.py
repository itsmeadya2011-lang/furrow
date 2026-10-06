from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

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
    task: asyncio.Task | None = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        queue: asyncio.Queue[object] = asyncio.Queue()
        done = object()

        def on_message(msg: str) -> None:
            queue.put_nowait(msg)

        client = LLMClient(settings=Settings(model=model)) if model else None
        orchestrator = Orchestrator(goal=goal, client=client, on_message=on_message)

        # Run the long-running orchestrator loop as a background task so it
        # does not block the WebSocket; its output is streamed through the
        # queue via the on_message callback.
        task = asyncio.create_task(orchestrator.run())
        task.add_done_callback(lambda _t: queue.put_nowait(done))

        while True:
            msg = await queue.get()
            if msg is done:
                break
            await websocket.send_text(str(msg))

        try:
            exc = task.exception()
        except asyncio.CancelledError:
            exc = None
        if exc is not None:
            await websocket.send_text(f"Error: {exc}")
        await websocket.send_text("\nRun finished.")
    except WebSocketDisconnect:
        pass
    finally:
        if task is not None and not task.done():
            task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
