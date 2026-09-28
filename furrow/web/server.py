from __future__ import annotations

import asyncio
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


HTML_PAGE = """<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
    <button id="stop" type="button" onclick="stop()">Stop</button>
  </form>
  <pre id="out" style="overflow:auto; max-height:80vh;"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const stopBtn = document.getElementById('stop');
    let ws = null;

    function append(text) {
      out.textContent += text + '\\n';
      out.scrollTop = out.scrollHeight;
    }

    function stop() {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      stopBtn.disabled = false;
      append('\\nStarting...\\n');
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => append(ev.data);
      ws.onerror = (ev) => append('\\n[error] ' + (ev.message || 'connection error') + '\\n');
      ws.onclose = () => { append('\\nClosed.\\n'); ws = null; };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[str] = asyncio.Queue()
    forward_task: asyncio.Task[Any] | None = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        def send_update(message: str) -> None:
            queue.put_nowait(message)

        async def forward() -> None:
            while True:
                message = await queue.get()
                try:
                    await websocket.send_text(message)
                except WebSocketDisconnect:
                    break
                queue.task_done()

        orchestrator = Orchestrator(goal=goal, output_callback=send_update)
        forward_task = asyncio.create_task(forward())
        await orchestrator.run()
        await queue.join()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(f"Error: {exc}")
        except Exception:
            pass
    finally:
        if forward_task is not None and not forward_task.done():
            forward_task.cancel()
            try:
                await forward_task
            except asyncio.CancelledError:
                pass
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: Optional[str] = None, port: Optional[int] = None) -> None:
    host = host or settings.host
    port = port or settings.port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
