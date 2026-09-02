from __future__ import annotations

import asyncio
import io
import json
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings
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


class WebSocketConsole(Console):
    def __init__(self, queue: asyncio.Queue[str]) -> None:
        super().__init__(file=io.StringIO(), force_terminal=True, soft_wrap=True)
        self._queue = queue

    def print(self, *objects: Any, **kwargs: Any) -> None:
        super().print(*objects, **kwargs)
        buf = self.file
        assert isinstance(buf, io.StringIO)
        text = buf.getvalue()
        if text:
            buf.seek(0)
            buf.truncate(0)
            try:
                self._queue.put_nowait(json.dumps({"type": "output", "text": text}))
            except Exception:
                pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    task: Optional[asyncio.Task[None]] = None
    queue: asyncio.Queue[str] = asyncio.Queue()
    ws_console = WebSocketConsole(queue)
    import furrow.core.orchestrator as orchestrator_module

    original_console = orchestrator_module.console
    orchestrator_module.console = ws_console
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        await websocket.send_text(json.dumps({"type": "start", "goal": goal}))
        orchestrator = Orchestrator(goal=goal)

        async def run_orchestrator() -> None:
            try:
                await orchestrator.run()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await queue.put(json.dumps({"type": "error", "message": str(exc)}))
            else:
                await queue.put(json.dumps({"type": "complete"}))

        task = asyncio.create_task(run_orchestrator())

        last_cycle = 0
        while not task.done():
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                if msg:
                    try:
                        payload = json.loads(msg)
                        if payload.get("type") == "stop":
                            task.cancel()
                            break
                    except Exception:
                        pass
            except asyncio.TimeoutError:
                pass
            while not queue.empty():
                try:
                    payload = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    await websocket.send_text(payload)
                except Exception:
                    task.cancel()
                    break
            if orchestrator.cycles > last_cycle:
                last_cycle = orchestrator.cycles
                try:
                    await websocket.send_text(
                        json.dumps({"type": "cycle", "num": last_cycle})
                    )
                except Exception:
                    break
        while not queue.empty():
            try:
                payload = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                await websocket.send_text(payload)
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        orchestrator_module.console = original_console
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
