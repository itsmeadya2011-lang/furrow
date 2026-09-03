from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

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
      ws.onmessage = (ev) => {
        try {
          const obj = JSON.parse(ev.data);
          out.textContent += JSON.stringify(obj, null, 2) + '\\n';
        } catch (err) {
          out.textContent += ev.data + '\\n';
        }
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onerror = () => out.textContent += '\\nError.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


async def _send(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload))


async def _run_with_events(ws: WebSocket, orchestrator: Orchestrator) -> None:
    async def forward(event: dict[str, Any]) -> None:
        await _send(ws, event)

    if hasattr(orchestrator, "on_event"):
        attr = orchestrator.on_event
        if callable(attr) and not inspect.iscoroutinefunction(attr):
            def sync_cb(event: dict[str, Any]) -> None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(forward(event))
                except RuntimeError:
                    pass
            orchestrator.on_event = sync_cb
        else:
            orchestrator.on_event = forward
        await orchestrator.run()
        return

    if hasattr(orchestrator, "stream"):
        stream = orchestrator.stream()
        if hasattr(stream, "__aiter__"):
            async for event in stream:
                await _send(ws, event)
            return
        for event in stream:
            await _send(ws, event)
        return

    await orchestrator.run()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)
        await _run_with_events(websocket, orchestrator)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await _send(websocket, {"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)