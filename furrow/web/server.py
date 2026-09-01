from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
from io import StringIO
from typing import Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient
import furrow.core.orchestrator as _orchestrator_module

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class _WSStream:
    def __init__(self, queue: asyncio.Queue[str]) -> None:
        self._queue = queue
        self._buf = ""

    def write(self, data: str) -> int:
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._queue.put_nowait(line)
        return len(data)

    def flush(self) -> None:
        if self._buf:
            self._queue.put_nowait(self._buf)
            self._buf = ""


def _build_client(model: str | None = None) -> LLMClient:
    settings = Settings()
    if model:
        settings = Settings(model=model)
    return LLMClient(settings=settings)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
    h1 { color: #333; }
    #form { margin: 20px 0; }
    input { padding: 8px; margin-right: 8px; }
    button { padding: 8px 16px; }
    #status { margin: 10px 0; font-weight: bold; }
    .running { color: orange; }
    .done { color: green; }
    .error { color: red; }
    #out { background: #f4f4f4; padding: 15px; border-radius: 4px; white-space: pre-wrap; font-family: monospace; min-height: 200px; max-height: 600px; overflow-y: auto; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required size="40" />
    <input id="model" placeholder="Model (optional)" />
    <button type="submit" id="btn">Start</button>
  </form>
  <div id="status"></div>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const status = document.getElementById('status');
    const btn = document.getElementById('btn');
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      const model = document.getElementById('model').value;
      status.textContent = 'Running...';
      status.className = 'running';
      btn.disabled = true;
      out.textContent = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => {
        ws.send(JSON.stringify({goal: goal, model: model || undefined}));
      };
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'log') {
          out.textContent += msg.data + '\\n';
          out.scrollTop = out.scrollHeight;
        } else if (msg.type === 'done') {
          status.textContent = 'Done: ' + msg.cycles + ' cycles';
          status.className = msg.passed ? 'done' : 'error';
          btn.disabled = false;
          ws.close();
        } else if (msg.type === 'error') {
          out.textContent += 'ERROR: ' + msg.data + '\\n';
          status.textContent = 'Error';
          status.className = 'error';
          btn.disabled = false;
        }
      };
      ws.onclose = () => {
        if (status.className !== 'done' && status.className !== 'error') {
          status.textContent = 'Disconnected';
          status.className = 'error';
          btn.disabled = false;
        }
      };
      ws.onerror = () => {
        status.textContent = 'Connection error';
        status.className = 'error';
        btn.disabled = false;
      };
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def sender() -> None:
        while True:
            line = await queue.get()
            if line is None:
                break
            try:
                await websocket.send_json({"type": "log", "data": line})
            except Exception:
                break

    sender_task = asyncio.create_task(sender())

    try:
        data = await websocket.receive_json()
        goal: str = data.get("goal", "")
        model: str | None = data.get("model") or None

        client = _build_client(model)
        orchestrator = Orchestrator(goal=goal, client=client)

        stream = _WSStream(queue)
        old_console = _orchestrator_module.console
        _orchestrator_module.console = Console(
            record=True,
            file=stream,
            force_terminal=False,
            color_system=None,
        )

        try:
            await orchestrator.run()
            passed = orchestrator._is_done()
            await websocket.send_json({
                "type": "done",
                "cycles": orchestrator.cycles,
                "passed": passed,
            })
        except Exception as exc:
            await websocket.send_json({"type": "error", "data": str(exc)})
            await websocket.send_json({
                "type": "done",
                "cycles": orchestrator.cycles,
                "passed": False,
            })
        finally:
            _orchestrator_module.console = old_console
    except WebSocketDisconnect:
        pass
    finally:
        queue.put_nowait(None)
        sender_task.cancel()


@app.post("/start")
async def start_endpoint(
    background_tasks: BackgroundTasks,
    request: StartRequest,
) -> dict[str, str]:
    background_tasks.add_task(_run_background, request.goal, request.model)
    return {"status": "started", "goal": request.goal}


async def _run_background(goal: str, model: str | None) -> None:
    client = _build_client(model)
    orchestrator = Orchestrator(goal=goal, client=client)
    await orchestrator.run()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
