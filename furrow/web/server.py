from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class ProgressEmitter:
    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def send(self, message: str) -> None:
        await self._websocket.send_text(message)

    async def drain(self) -> None:
        while True:
            msg = await self._queue.get()
            await self.send(msg)

    def print(self, message: str) -> None:
        self._queue.put_nowait(message)


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
    <input id="model" placeholder="Model (optional)" />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      const goal = document.getElementById('goal').value;
      const model = document.getElementById('model').value;
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onerror = (ev) => out.textContent += '\\nError: ' + ev.type + '\\n';
      ws.send(JSON.stringify({goal, model}));
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
        emitter = ProgressEmitter(websocket)
        drain_task = asyncio.create_task(emitter.drain())
        try:
            settings = Settings()
            if model:
                settings.model = model
            orchestrator = Orchestrator(goal=goal, settings=settings)
            # Monkey-patch console for streaming
            from furrow.core import orchestrator as orch_mod
            original_print = orch_mod.console.print
            def streaming_print(*args, **kwargs):
                text = " ".join(str(a) for a in args)
                emitter.print(text)
                original_print(*args, **kwargs)
            orch_mod.console.print = streaming_print
            try:
                await orchestrator.run()
            finally:
                orch_mod.console.print = original_print
        finally:
            drain_task.cancel()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"[red]Error: {e}[/red]")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
