from __future__ import annotations

import asyncio
import io
import re

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
    model: str | None = None


class WebSocketStream:
    """File-like object that buffers Rich Console output and sends to WebSocket."""

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self._buffer = io.StringIO()

    def write(self, s: str) -> int:
        self._buffer.write(s)
        return len(s)

    def flush(self) -> None:
        content = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate(0)
        if content and content.strip():
            clean = self._ANSI_RE.sub("", content)
            loop = asyncio.get_running_loop()
            loop.create_task(self._send(clean))

    async def _send(self, content: str) -> None:
        try:
            await self.websocket.send_text(content)
        except WebSocketDisconnect:
            pass

    def isatty(self) -> bool:
        return False


def make_streaming_console(websocket: WebSocket) -> Console:
    """Create a Rich Console that streams output to the WebSocket."""
    return Console(
        file=WebSocketStream(websocket),
        force_terminal=False,
        no_color=True,
        width=120,
    )


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; margin: 20px; background: #1a1a2e; color: #e0e0e0; }
  h1 { color: #4CAF50; }
  #out { background: #16213e; padding: 10px; border-radius: 5px; white-space: pre-wrap; }
  form { margin-bottom: 10px; }
  input { padding: 5px; width: 400px; background: #0f3460; color: #fff; }
  button { padding: 5px 15px; background: #4CAF50; color: #fff; border: none; cursor: pointer; }
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
        streaming_console = make_streaming_console(websocket)
        web_settings = Settings()
        orchestrator = Orchestrator(
            goal=goal, console=streaming_console, settings=web_settings
        )
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await websocket.send_text(f"\nFurrow error: {e}\n")
        except Exception:  # noqa: S110, BLE001
            pass
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: S110, BLE001
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
