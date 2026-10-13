from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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


# Orchestrator output is streamed as one line per message. The orchestrator
# itself only knows how to call a *synchronous* callback, but sending over a
# WebSocket must happen on the event loop. So the callback just pushes lines onto
# a queue and a concurrent drainer task forwards them to the client. This keeps
# the callback non-blocking and lets us stop the orchestrator the instant the
# client disconnects. A `None` value on the queue signals completion.


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    goal = data.get("goal", "")

    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    stopped = asyncio.Event()
    runner: asyncio.Task[None] | None = None
    drainer: asyncio.Task[None] | None = None
    listener: asyncio.Task[None] | None = None

    def emit(line: str) -> None:
        if not stopped.is_set():
            queue.put_nowait(line)

    orchestrator = Orchestrator(goal=goal, output_callback=emit)

    async def run_orchestrator() -> None:
        try:
            await orchestrator.run()
        except Exception as exc:
            # Surface crashes to the client instead of killing the socket.
            emit(f"Orchestrator stopped: {exc}")
        finally:
            if not stopped.is_set():
                queue.put_nowait(None)

    async def drain() -> None:
        try:
            while True:
                item = await queue.get()
                if item is None:
                    await websocket.send_text("DONE")
                    break
                await websocket.send_text(item)
        except Exception:
            # Any send/get failure means the client is gone; stop everything.
            pass
        finally:
            stopped.set()
            if runner is not None:
                runner.cancel()

    async def listen() -> None:
        # The client only sends once, so this blocks until the connection
        # drops, giving prompt awareness of a mid-run disconnect.
        try:
            while True:
                await websocket.receive()
        except Exception:
            # Client disconnected (or errored on the socket).
            pass
        finally:
            stopped.set()
            if runner is not None:
                runner.cancel()

    runner = asyncio.create_task(run_orchestrator())
    drainer = asyncio.create_task(drain())
    listener = asyncio.create_task(listen())

    try:
        await runner
    except asyncio.CancelledError:
        pass
    finally:
        stopped.set()
        for task in (runner, drainer, listener):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(runner, drainer, listener, return_exceptions=True)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
