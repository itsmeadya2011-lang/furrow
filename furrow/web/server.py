from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import suppress
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)
POLL_SECONDS = 0.5

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
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const show = (text) => { out.textContent += text + '\\n'; };
    function render(ev) {
      let m;
      try { m = JSON.parse(ev.data); } catch (err) { return show(ev.data); }
      if (m.type === 'accepted') show('Running: ' + m.goal + ' [' + m.model + ']');
      else if (m.type === 'progress') show('  cycle ' + m.cycle + ': ' + m.goal);
      else if (m.type === 'complete') show('Done after ' + m.cycles + ' cycle(s).');
      else if (m.type === 'error') show('Error: ' + m.error);
      else show(ev.data);
    }
    form.onsubmit = async (e) => {
      e.preventDefault();
      show('Starting...');
      const ws = new WebSocket(scheme + '://' + location.host + '/ws');
      ws.onmessage = render;
      ws.onclose = () => show('Closed.');
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


async def _drive(websocket: WebSocket, goal: str, config: Settings) -> None:
    """Run the orchestrator, pushing a message per cycle, then a terminal message."""
    orchestrator = Orchestrator(goal=goal, settings=config)
    task: asyncio.Task[None] = asyncio.create_task(orchestrator.run())
    cycles = 0
    try:
        while not task.done():
            await asyncio.sleep(POLL_SECONDS)
            # The orchestrator owns the loop, so poll its cycle counter for progress.
            current = getattr(orchestrator, "cycles", 0)
            if current != cycles:
                cycles = current
                await websocket.send_json(
                    {"type": "progress", "cycle": cycles, "goal": orchestrator.goal}
                )
        await task
    except asyncio.CancelledError:
        task.cancel()  # client went away mid-run; do not leave the loop spinning
        raise
    except Exception as exc:
        logger.exception("orchestrator failed for goal %r", goal)
        await websocket.send_json({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
    else:
        await websocket.send_json(
            {"type": "complete", "cycles": getattr(orchestrator, "cycles", cycles)}
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    runner: asyncio.Task[None] | None = None
    try:
        try:
            request = StartRequest.model_validate(await websocket.receive_json())
        except WebSocketDisconnect:
            logger.info("websocket closed before a goal was sent")
            return
        except json.JSONDecodeError as exc:
            await websocket.send_json({"type": "error", "error": f"malformed JSON: {exc}"})
            return
        except ValidationError as exc:
            problems = "; ".join(str(error["msg"]) for error in exc.errors())
            await websocket.send_json({"type": "error", "error": f"invalid request: {problems}"})
            return
        config = settings.model_copy(update={"model": request.model}) if request.model else settings
        await websocket.send_json({"type": "accepted", "goal": request.goal, "model": config.model})
        runner = asyncio.create_task(_drive(websocket, request.goal, config))
        await runner
    except WebSocketDisconnect:
        logger.info("websocket disconnected; stopping the orchestrator")
    finally:
        if runner is not None and not runner.done():
            runner.cancel()
        # RuntimeError covers WebSocketDisconnected / already-closed sends.
        with suppress(WebSocketDisconnect, RuntimeError):
            await websocket.close()


def run(host: str | None = None, port: int | None = None) -> None:
    # 0.0.0.0 binds every interface, so the default UI is reachable from outside the
    # machine. config.py owns no web fields, so host/port come from here and the
    # FURROW_WEB_HOST / FURROW_WEB_PORT env vars, which match its FURROW_ prefix.
    host = host or os.getenv("FURROW_WEB_HOST", "0.0.0.0")
    port = port or int(os.getenv("FURROW_WEB_PORT", "8000"))
    logger.info("serving Furrow on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)
