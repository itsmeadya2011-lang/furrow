from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")

_orchestrator_tasks: dict[str, asyncio.Task] = {}


class StartRequest(BaseModel):
    goal: str
    provider: Provider | None = None
    model: str | None = None
    max_cycles: int | None = None
    max_parallel_tasks: int | None = None
    log_level: str | None = None
    test_command: str | None = None
    test_timeout: int | None = None
    workspace: str | None = None


def _build_settings(base: Settings, request: StartRequest) -> Settings:
    settings = base.model_copy(deep=True)
    if request.model:
        settings.model = request.model
        settings.worker_model = request.model
        settings.planner_model = request.model
        settings.tester_model = request.model
    if request.provider:
        settings.provider = request.provider
    if request.max_cycles is not None:
        settings.max_cycles = request.max_cycles
    if request.max_parallel_tasks is not None:
        settings.max_parallel_tasks = request.max_parallel_tasks
    if request.log_level:
        settings.log_level = request.log_level
    if request.test_command:
        settings.test_command = request.test_command
    if request.test_timeout is not None:
        settings.test_timeout = request.test_timeout
    if request.workspace:
        settings.workspace = request.workspace
    return settings


async def start_orchestrator(request: StartRequest) -> None:
    """Run orchestrator for a request (used for background tasks)."""
    _configure_logging(request.log_level or "INFO")
    base_settings = Settings()
    settings = _build_settings(base_settings, request)
    client = LLMClient(settings=settings)
    orchestrator = Orchestrator(goal=request.goal, client=client)
    await orchestrator.run()


def _configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname)s (%(name)s) %(message)s",
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/start")
async def start(request: StartRequest) -> dict[str, str]:
    task_id = str(uuid.uuid4())
    task = asyncio.create_task(start_orchestrator(request))
    _orchestrator_tasks[task_id] = task
    # Remove task when done to prevent memory leak
    task.add_done_callback(lambda t: _orchestrator_tasks.pop(task_id, None))
    return {"task_id": task_id}


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
    <input id="model" placeholder="Model override (optional)" />
    <input id="max_cycles" type="number" placeholder="Max cycles (0 = unlimited)" />
    <input id="max_parallel_tasks" type="number" placeholder="Max parallel tasks" />
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
      ws.send(JSON.stringify({
        goal: document.getElementById('goal').value,
        model: document.getElementById('model').value || undefined,
        max_cycles: parseInt(document.getElementById('max_cycles').value) || undefined,
        max_parallel_tasks: parseInt(document.getElementById('max_parallel_tasks').value) || undefined
      }));
    };
  </script>
</body>
</html>
"""
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        raw = await websocket.receive_json()
    except Exception:  # noqa: BLE001
        return

    request = StartRequest(**raw)
    _configure_logging(request.log_level or "INFO")
    base_settings = Settings()
    settings = _build_settings(base_settings, request)
    client = LLMClient(settings=settings)

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def on_log(event: str, data: dict[str, Any]) -> None:
        message = json.dumps({"event": event, **data})
        queue.put_nowait(message)

    async def runner() -> None:
        try:
            await Orchestrator(goal=request.goal, client=client, on_log=on_log).run()
        except Exception as e:  # noqa: BLE001
            on_log("orchestrator_error", {"error": str(e), "traceback": repr(e)})
        finally:
            queue.put_nowait(None)

    async def streamer() -> None:
        while True:
            message = await queue.get()
            if message is None:
                return
            await websocket.send_text(message)

    runner_task = asyncio.create_task(runner())
    streamer_task = asyncio.create_task(streamer())
    try:
        await asyncio.gather(runner_task, streamer_task)
    except WebSocketDisconnect:
        runner_task.cancel()
        await asyncio.gather(runner_task, return_exceptions=True)
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
