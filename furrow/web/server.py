from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import settings as default_settings
from furrow.core.orchestrator import Orchestrator

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
      ws.onerror = (ev) => out.textContent += '\\nError: ' + ev + '\\n';
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
    queue: asyncio.Queue = asyncio.Queue()

    def on_output(message: str) -> None:
        queue.put_nowait(message)

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        settings = default_settings
        if data.get("model"):
            settings.model = data["model"]

        orchestrator = Orchestrator(goal=goal, settings=settings)
        orchestrator.on_output = on_output

        # Run orchestrator in background task
        run_task = asyncio.create_task(orchestrator.run())

        # Stream output to client while orchestrator runs
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=0.5)
                await websocket.send_text(message)
            except asyncio.TimeoutError:
                if run_task.done():
                    break

        # Drain remaining messages
        while not queue.empty():
            message = queue.get_nowait()
            await websocket.send_text(message)

        # Check for exceptions
        if run_task.exception():
            await websocket.send_text(f"\n[ERROR] {run_task.exception()}\n")

        await websocket.send_text("\n[Furrow] Done.\n")

    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await websocket.send_text(f"\n[ERROR] {e}\n")
        except Exception:  # noqa: BLE001, S110
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
