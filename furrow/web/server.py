from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator
from furrow.logging import get_logger

app = FastAPI(title="Furrow")
_logger = get_logger("web")


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
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        # Queue to collect live output from the orchestrator
        output_queue: asyncio.Queue[str] = asyncio.Queue()

        def output_callback(message: str) -> None:
            output_queue.put_nowait(message)

        orchestrator = Orchestrator(goal=goal, output_callback=output_callback)

        async def forward_output() -> None:
            """Drain the output queue and send messages to the WebSocket."""
            while True:
                try:
                    message = await asyncio.wait_for(output_queue.get(), timeout=0.5)
                    await websocket.send_text(message)
                except asyncio.TimeoutError:
                    # Break only when orchestrator is done AND queue is drained
                    if orchestrator_task.done() and output_queue.empty():
                        break

        orchestrator_task = asyncio.create_task(orchestrator.run())
        forwarder_task = asyncio.create_task(forward_output())

        try:
            await orchestrator_task
        except Exception as e:
            _logger.error("Orchestrator failed", error=str(e))
            await websocket.send_text(f"ERROR: {e}")
        finally:
            forwarder_task.cancel()
            try:
                await forwarder_task
            except asyncio.CancelledError:
                pass

        await websocket.close()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
