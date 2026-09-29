from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class WebSocketConsole(Console):
    """A rich Console subclass that streams output to a WebSocket connection."""

    def __init__(self, websocket: WebSocket, **kwargs) -> None:
        super().__init__(**kwargs)
        self._websocket = websocket

    async def _async_send(self, text: str) -> None:
        await self._websocket.send_text(text)

    def print(self, *objects, **kwargs) -> None:  # type: ignore[override]
        """Render output to text and stream it to the WebSocket."""
        from io import StringIO

        buffer = StringIO()
        capture_console = Console(file=buffer, force_terminal=False, width=self.width)
        capture_console.print(*objects, **kwargs)
        text = buffer.getvalue()
        if text.strip():
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_send(text))
            except RuntimeError:
                pass


def _make_ws_console(websocket: WebSocket) -> WebSocketConsole:
    return WebSocketConsole(websocket, force_terminal=False, width=120)


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
      ws.onmessage = (ev) => out.textContent += ev.data;
      ws.onclose = () => out.textContent += '\\nDone.\\n';
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
        model = data.get("model")
        if model:
            from furrow.config import settings
            settings.model = model
            settings.planner_model = model
            settings.worker_model = model
            settings.tester_model = model

        ws_console = _make_ws_console(websocket)
        orchestrator = Orchestrator(goal=goal, console=ws_console)
        await orchestrator.run()
        await websocket.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001 — relay any error to the client
        await websocket.send_text(f"\n[red]Error: {e}[/red]\n")
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
