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


class WebSocketSink:
    """OutputSink implementation that sends text to a WebSocket."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    async def emit(self, text: str) -> None:
        await self._ws.send_text(text)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: monospace; margin: 2rem; background: #1a1a2e; color: #e0e0e0; }
  h1 { color: #4CAF50; }
  input { width: 60vw; padding: .5rem; background: #16213e; color: #e0e0e0; border: 1px solid #0f3460; }
  button { padding: .5rem 1rem; background: #4CAF50; color: #fff; border: none; cursor: pointer; }
  button:hover { background: #45a049; }
  pre#out { background: #0f0f23; padding: 1rem; border-radius: 8px; height: 70vh; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }
</style>
</head>
<body>
  <h1>Furrow — Autonomous Coding Agent</h1>
  <form id="form">
    <input id="goal" placeholder="Enter your goal..." required />
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
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\n--- Connection closed ---\\n';
      ws.onerror = () => out.textContent += '\\n--- Connection error ---\\n';
      ws.onopen = () => {
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
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
        if not goal:
            await websocket.send_json(
                {"type": "error", "message": "No goal provided"}
            )
            await websocket.close()
            return

        await websocket.send_json(
            {"type": "info", "message": f"Starting Furrow: {goal}"}
        )

        sink = WebSocketSink(websocket)
        orchestrator = Orchestrator(goal=goal, output=sink)

        await orchestrator.run()

        await websocket.close()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        finally:
            await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
