from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

_orchestrator_state: dict[str, Any] = {
    "running": False,
    "goal": None,
    "cycles": 0,
}


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "healthy", "running": _orchestrator_state["running"]}


@app.get("/status")
async def status() -> dict[str, Any]:
    return dict(_orchestrator_state)


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
    orchestrator: Orchestrator | None = None
    run_task: asyncio.Task[None] | None = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)
        _orchestrator_state.update({"running": True, "goal": goal, "cycles": 0})

        async def send_progress(state: dict[str, Any]) -> None:
            _orchestrator_state.update(state)
            await websocket.send_json(state)

        max_cycles = settings.max_cycles or None
        run_task = asyncio.create_task(
            orchestrator.run(max_cycles=max_cycles, status_callback=send_progress)
        )
        await asyncio.shield(run_task)
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {
                    "type": "complete",
                    "goal": goal,
                    "cycles": orchestrator.cycles if orchestrator is not None else 0,
                }
            )
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "error": repr(exc)})
    finally:
        if run_task is not None and not run_task.done():
            run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await run_task
        if orchestrator is not None:
            _orchestrator_state["running"] = False


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
