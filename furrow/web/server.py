from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


def _configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


_configure_logging()
log = structlog.get_logger("furrow.web")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    planner_model: str | None = None
    worker_model: str | None = None
    tester_model: str | None = None
    provider: str | None = None
    max_cycles: int | None = None
    max_parallel: int | None = None
    workspace: str | None = None


def _apply_settings(req: StartRequest) -> None:
    if req.model:
        settings.model = req.model
        settings.planner_model = req.model
        settings.worker_model = req.model
        settings.tester_model = req.model
    if req.planner_model:
        settings.planner_model = req.planner_model
    if req.worker_model:
        settings.worker_model = req.worker_model
    if req.tester_model:
        settings.tester_model = req.tester_model
    if req.provider:
        settings.provider = Provider(req.provider)
    if req.max_cycles is not None:
        settings.max_cycles = req.max_cycles
    if req.max_parallel is not None:
        settings.max_parallel_tasks = req.max_parallel
    if req.workspace:
        settings.workspace = Path(req.workspace)


INDEX_HTML = """<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <input id="model" placeholder="Model (optional, sets all roles)" />
    <input id="max-cycles" type="number" min="0" placeholder="Max cycles (0=infinite)" />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      out.textContent += 'Starting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => { out.textContent += ev.data + '\\n'; out.scrollTop = out.scrollHeight; };
      ws.onerror = () => { out.textContent += '[websocket error]\\n'; };
      ws.onclose = () => { out.textContent += '\\nDone.\\n'; };
      const payload = { goal: document.getElementById('goal').value };
      const model = document.getElementById('model').value;
      if (model) payload.model = model;
      const mc = document.getElementById('max-cycles').value;
      if (mc !== '') payload.max_cycles = parseInt(mc, 10);
      ws.onopen = () => ws.send(JSON.stringify(payload));
    };
  </script>
</body>
</html>
"""


async def _drain_messages(websocket: WebSocket, queue: asyncio.Queue[str | None]) -> None:
    while True:
        msg = await queue.get()
        if msg is None:
            break
        try:
            await websocket.send_text(msg)
        except Exception:
            log.warning("websocket send failed", exc_info=True)
            break


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    drain_task = asyncio.create_task(_drain_messages(websocket, queue))

    def on_message(msg: str) -> None:
        queue.put_nowait(msg)

    async def teardown() -> None:
        await queue.put(None)
        try:
            await asyncio.wait_for(drain_task, timeout=5)
        except asyncio.TimeoutError:
            drain_task.cancel()

    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        await teardown()
        await websocket.close()
        return
    except Exception as exc:  # noqa: BLE001
        await queue.put(f"[ERROR] Invalid request: {exc}")
        await teardown()
        try:
            await websocket.close(code=1008)
        except Exception:
            log.warning("websocket close failed", exc_info=True)
        return

    if not isinstance(data, dict):
        await queue.put("[ERROR] Invalid request: expected a JSON object")
        await teardown()
        try:
            await websocket.close(code=1008)
        except Exception:
            log.warning("websocket close failed", exc_info=True)
        return

    try:
        request = StartRequest(**data)
    except ValidationError as exc:
        await queue.put(f"[ERROR] Invalid request: {exc}")
        await teardown()
        try:
            await websocket.close(code=1008)
        except Exception:
            log.warning("websocket close failed", exc_info=True)
        return

    _apply_settings(request)
    cfg = {
        k: v
        for k, v in request.model_dump(mode="json").items()
        if v is not None
    }
    log.info("furrow ws start", **cfg)

    try:
        orchestrator = Orchestrator(goal=request.goal, on_message=on_message)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.exception("orchestrator failed", error=str(exc))
        await queue.put(f"[ERROR] {exc}")
    finally:
        await teardown()
        try:
            await websocket.close()
        except Exception:
            log.warning("websocket close failed", exc_info=True)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
