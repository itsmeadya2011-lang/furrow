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


class _WebSocketFile:
    """File-like object that sends output to a WebSocket via the event loop."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    def write(self, data: str) -> int:
        if data.strip():
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._websocket.send_text(data))
            except RuntimeError:
                pass
        return len(data)

    def flush(self) -> None:
        pass


class WebSocketConsole(Console):
    """A Rich Console that streams output to a WebSocket connection."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        super().__init__(
            file=_WebSocketFile(websocket),
            force_terminal=False,
            no_color=True,
            width=120,
            legacy_windows=False,
        )

    @property
    def is_terminal(self) -> bool:
        return False


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
      out.textContent = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
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
    ws_console = WebSocketConsole(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal, console=ws_console)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
