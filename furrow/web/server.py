from __future__ import annotations

import asyncio
from contextlib import suppress

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


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
"""
    )


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

        async def send_progress(message: str) -> None:
            with suppress(WebSocketDisconnect):
                await websocket.send_text(message)

        orchestrator = Orchestrator(goal=goal, progress_callback=send_progress)

        # Run the orchestrator, periodically flushing progress
        task = asyncio.create_task(orchestrator.run())

        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            await send_progress(f"[error] {e}")

        await websocket.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        with suppress(WebSocketDisconnect):
            await websocket.send_text(f"[error] {e}")
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
