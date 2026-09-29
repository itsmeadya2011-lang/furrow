from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


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

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def drain() -> None:
        """Background task that pushes queued messages to the client."""
        while True:
            msg = await queue.get()
            if msg is None:
                return
            try:
                await websocket.send_text(msg)
            except WebSocketDisconnect:
                return

    drain_task = asyncio.create_task(drain())
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)

        def emit(msg: str) -> None:
            """Wrap queue.put_nowait so QueueFull never raises into the orchestrator."""
            try:
                queue.put_nowait(msg)
            except asyncio.QueueFull:
                pass

        await orchestrator.run(output_callback=emit)

        # Final marker, ordered after every orchestrator message.
        queue.put_nowait("[DONE]")
        # Sentinel that tells the drain task to stop once everything is flushed.
        queue.put_nowait(None)
        await drain_task
        # Close the connection so clients know the stream is finished.
        await websocket.close()
    except WebSocketDisconnect:
        pass
    finally:
        if not drain_task.done():
            drain_task.cancel()
            try:
                await drain_task
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
