from __future__ import annotations

import asyncio
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console
from rich.highlighter import NullHighlighter

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketConsole(Console):
    """A Rich Console subclass that sends output to a WebSocket."""

    def __init__(self, websocket: WebSocket) -> None:
        super().__init__(highlighter=NullHighlighter(), markup=False)
        self.websocket = websocket

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Render the output and send it to the WebSocket."""
        with self.capture() as capture:
            super().print(*args, **kwargs)
        text = capture.get()
        if text.strip():
            # Fire and forget - don't block on websocket errors
            asyncio.create_task(self._send(text))

    async def _send(self, text: str) -> None:
        try:
            await self.websocket.send_text(text)
        except Exception:
            pass  # WebSocket closed


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
        ws_console = WebSocketConsole(websocket)
        orchestrator = Orchestrator(goal=goal, output_console=ws_console)
        await orchestrator.run()
        await websocket.send_text("\n[Furrow finished]")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"\nError: {e}")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
