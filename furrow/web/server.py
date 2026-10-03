from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Background runs started via POST /start, kept referenced so they are not GC'd.
_BACKGROUND_TASKS: set[asyncio.Task] = set()

WS_POLICY_VIOLATION = 1008


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


def settings_for(model: str | None) -> Settings:
    """Return the settings for a run.

    A truthy *model* yields a copy with ``model`` overridden, so the singleton is
    never mutated. When *model* is falsy the singleton itself is returned unchanged.
    """
    if model:
        return settings.model_copy(update={"model": model})
    return settings


def make_on_event(websocket: WebSocket) -> Callable[[dict], Awaitable[None]]:
    """Build an on_event callback that streams newline-terminated JSON.

    Send failures are swallowed so a dead socket can never kill the run.
    """

    async def on_event(event: dict) -> None:
        try:
            await websocket.send_text(json.dumps(event, default=str) + "\n")
        except Exception:
            pass

    return on_event


async def wait_for_disconnect(websocket: WebSocket) -> None:
    """Complete (or raise) as soon as the peer goes away."""
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))


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
        message = await websocket.receive()
    except WebSocketDisconnect:
        return
    if message.get("type") == "websocket.disconnect":
        return

    raw: Any = message.get("text")
    if raw is None:
        raw = (message.get("bytes") or b"").decode("utf-8", "replace")

    try:
        request = StartRequest.model_validate_json(raw)
        goal = request.goal.strip()
        if not goal:
            raise ValueError("goal must not be empty")
    except ValueError as exc:
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": str(exc)}) + "\n"
            )
            await websocket.close(code=WS_POLICY_VIOLATION)
        except Exception:
            pass
        return

    orchestrator = Orchestrator(
        goal=goal,
        settings=settings_for(request.model),
        on_event=make_on_event(websocket),
    )

    run_task = asyncio.create_task(orchestrator.run())
    recv_task = asyncio.create_task(wait_for_disconnect(websocket))
    try:
        await asyncio.wait(
            {run_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        # A client disconnect must terminate the run, and vice versa.
        for task in (run_task, recv_task):
            task.cancel()
        await asyncio.gather(run_task, recv_task, return_exceptions=True)

    try:
        await websocket.close()
    except Exception:
        pass


@app.post("/start")
async def start(request: StartRequest) -> dict:
    goal = request.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="goal must not be empty")
    orchestrator = Orchestrator(goal=goal, settings=settings_for(request.model))
    task = asyncio.create_task(orchestrator.run())
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return {"status": "started", "goal": goal}


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)