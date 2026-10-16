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
    queue: asyncio.Queue[str] = asyncio.Queue()
    run_task: asyncio.Task[None] | None = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal, output_queue=queue)
        run_task = asyncio.create_task(orchestrator.run())

        while True:
            done, pending = await asyncio.wait(
                [
                    run_task,
                    asyncio.create_task(queue.get()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task is run_task:
                    for remaining in pending:
                        remaining.cancel()
                    break
                else:
                    msg = task.result()
                    if msg:
                        await websocket.send_text(msg)
            else:
                continue
            break

        # Drain any remaining messages
        while not queue.empty():
            msg = await queue.get()
            if msg:
                await websocket.send_text(msg)

        await websocket.send_text("\nDone.")
    except WebSocketDisconnect:
        if run_task is not None and not run_task.done():
            run_task.cancel()
    except Exception:  # noqa: S110, BLE001 - intentionally swallow errors at socket boundary
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
