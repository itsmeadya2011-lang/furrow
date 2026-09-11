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
        orchestrator = Orchestrator(goal=goal)

        # Start the orchestrator background task *before* any sends so it is
        # always covered by the cleanup block below. This prevents a leak if
        # the client disconnects during the initial progress send.
        run_task = asyncio.create_task(orchestrator.run())

        async def _cleanup(task: asyncio.Task) -> None:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                # The task finished with an exception that has already been
                # reported by the handler above; suppress it here so it does
                # not propagate into the outer error handler.
                pass

        try:
            await websocket.send_json(
                {"type": "progress", "message": "Starting orchestration..."}
            )

            # Run the orchestrator as a background task so we can emit periodic
            # progress while it works. The orchestrator exposes no callbacks, so
            # we surface its live ``cycles`` counter as a meaningful progress value.
            while not run_task.done():
                await websocket.send_json(
                    {
                        "type": "progress",
                        "message": "Working",
                        "cycles": orchestrator.cycles,
                    }
                )
                await asyncio.sleep(5.0)

            if run_task.cancelled():
                return

            exc = run_task.exception()
            if exc is not None:
                await websocket.send_json(
                    {"type": "error", "message": str(exc)}
                )
            else:
                await websocket.send_json(
                    {
                        "type": "complete",
                        "message": "Orchestration finished",
                        "cycles": orchestrator.cycles,
                    }
                )
        finally:
            await _cleanup(run_task)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await websocket.send_json(
            {"type": "error", "message": str(exc)}
        )


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
