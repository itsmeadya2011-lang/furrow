from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketOutputStream:
    """A minimal file-like object that sends captured output to a WebSocket.

    Rich's ``Console`` writes text to its ``file`` — by plugging this stream
    in we get every ``console.print(...)`` call from the orchestrator relayed
    back to the browser in real time.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    async def _safe_send(self, text: str) -> None:
        try:
            await self._ws.send_text(text)
        except WebSocketDisconnect:
            pass

    def write(self, text: str) -> int:
        if text and text.strip():
            asyncio.create_task(self._safe_send(text))
        return len(text)

    def flush(self) -> None:
        pass


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<meta charset="utf-8">
<style>
  body { font-family: monospace; margin: 2em; background: #1a1a2e; color: #eee; }
  h1 { color: #4CAF50; }
  #out { white-space: pre-wrap; background: #111; padding: 1em; border-radius: 4px; height: 500px; overflow-y: auto; }
  input { width: 60%; padding: 0.5em; }
  button { padding: 0.5em 1em; background: #4CAF50; color: white; border: none; cursor: pointer; }
</style>
</head>
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
      ws.onmessage = (ev) => out.textContent += ev.data;
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onopen = () => ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
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

        # Build a console that streams every Rich print directly to the client.
        # Plain text output (no ANSI codes) — ideal for rendering in a browser.
        ws_console = Console(
            file=WebSocketOutputStream(websocket),
            color_system=None,
            width=120,
        )
        orchestrator = Orchestrator(goal=goal, console=ws_console)
        if model:
            from furrow.config import settings
            settings.model = model
            orchestrator.client.settings.model = model

        # Run the orchestrator — all console output is streamed via ws_console.
        await orchestrator.run()
        await websocket.send_text("\n[done]\n")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"\n[error] {e}\n")
    finally:
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
