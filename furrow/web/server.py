from __future__ import annotations

import asyncio
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
      ws.send(JSON.stringify({goal: document.getElementById('goal').value, model: null}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    orch_task: Optional[asyncio.Task] = None
    consume_task: Optional[asyncio.Task] = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        queue: asyncio.Queue[str] = asyncio.Queue()

        async def consume() -> None:
            while True:
                msg = await queue.get()
                await websocket.send_text(msg)
                queue.task_done()

        orchestrator = Orchestrator(
            goal=goal,
            on_output=lambda msg: queue.put_nowait(msg),
        )

        orch_task = asyncio.ensure_future(orchestrator.run())
        consume_task = asyncio.ensure_future(consume())

        done, pending = await asyncio.wait(
            {orch_task, consume_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await websocket.send_text("[Furrow] Run complete.")
    except WebSocketDisconnect:
        if orch_task and not orch_task.done():
            orch_task.cancel()
        if consume_task and not consume_task.done():
            consume_task.cancel()
    except Exception as e:
        try:
            await websocket.send_text(f"[Furrow] Error: {e}")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
