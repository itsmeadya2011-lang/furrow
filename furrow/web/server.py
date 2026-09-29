from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Plan
from furrow.config import settings as default_settings
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
    stop = asyncio.Event()

    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return
    goal: str = data.get("goal", "")
    model: Optional[str] = data.get("model")

    settings = default_settings
    if model is not None:
        settings = default_settings.model_copy(
            update={
                "model": model,
                "planner_model": model,
                "worker_model": model,
                "tester_model": model,
            }
        )

    orchestrator = Orchestrator(goal=goal, settings=settings)

    async def on_cycle_end(cycle: int, plan: Plan) -> None:
        try:
            await websocket.send_json(
                {
                    "type": "cycle",
                    "cycle": cycle,
                    "goal": goal,
                    "tasks": [task.model_dump() for task in plan.tasks],
                    "rationale": plan.rationale,
                }
            )
        except (WebSocketDisconnect, RuntimeError):
            stop.set()

    run_task = asyncio.create_task(orchestrator.run(on_cycle_end=on_cycle_end))

    async def watch_disconnect() -> None:
        try:
            while not stop.is_set():
                await websocket.receive()
        except WebSocketDisconnect:
            pass
        finally:
            stop.set()
            if not run_task.done():
                run_task.cancel()

    watcher = asyncio.create_task(watch_disconnect())

    try:
        await run_task
    except asyncio.CancelledError:
        pass
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
    else:
        try:
            await websocket.send_json(
                {"type": "done", "cycles": orchestrator.cycles}
            )
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
    finally:
        stop.set()
        if not watcher.done():
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
        if not run_task.done():
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
