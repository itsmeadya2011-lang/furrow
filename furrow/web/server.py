from __future__ import annotations

import asyncio
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StreamingConsole(Console):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._messages = asyncio.Queue()

    def print(self, *args: Any, **kwargs: Any) -> None:
        with self.capture() as capture:
            super().print(*args, **kwargs)
        text = capture.get()
        self._messages.put_nowait(text)


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
        model = data.get("model")

        if model:
            from furrow.config import settings
            settings.model = model

        streaming_console = StreamingConsole()
        orchestrator = Orchestrator(goal=goal)
        orchestrator.console = streaming_console

        task = asyncio.create_task(orchestrator.run())
        try:
            while not task.done() or not streaming_console._messages.empty():
                try:
                    message = await asyncio.wait_for(
                        streaming_console._messages.get(), timeout=0.1
                    )
                    await websocket.send_text(message)
                except asyncio.TimeoutError:
                    if task.done():
                        break

            try:
                task.result()
            except asyncio.CancelledError:
                await websocket.send_text("cancelled")
            except Exception as exc:
                await websocket.send_text(f"failed: {exc}")
            else:
                await websocket.send_text("done")
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
