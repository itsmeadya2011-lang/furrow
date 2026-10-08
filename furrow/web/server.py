from __future__ import annotations

import asyncio
import io
from typing import Any, Optional

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
    model: Optional[str] = None


class WebSocketConsole(Console):
    """A rich Console that also streams print() output to a WebSocket."""

    def __init__(self, websocket: WebSocket, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.websocket = websocket
        self._pending: set[asyncio.Task[None]] = set()

    def print(self, *args: Any, **kwargs: Any) -> None:
        super().print(*args, **kwargs)
        text = self._render(*args, **kwargs)
        if text:
            self._schedule_send(text)

    def _render(self, *args: Any, **kwargs: Any) -> str:
        buffer = io.StringIO()
        capture = Console(
            file=buffer,
            width=self.width,
            no_color=True,
            legacy_windows=False,
            force_terminal=False,
        )
        capture.print(*args, **kwargs)
        return buffer.getvalue().rstrip("\n")

    def _schedule_send(self, text: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._send(text))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _send(self, text: str) -> None:
        try:
            await self.websocket.send_text(text)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    async def wait(self) -> None:
        """Flush all outstanding send tasks so no streamed output is dropped."""
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)


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
    <button id="start" type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const goalInput = document.getElementById('goal');
    const startBtn = document.getElementById('start');
    const out = document.getElementById('out');
    let ws = null;

    function appendLine(text) {
      out.textContent += text + '\\n';
      out.scrollTop = out.scrollHeight;
    }

    form.onsubmit = (e) => {
      e.preventDefault();
      if (ws && ws.readyState === WebSocket.OPEN) return;
      startBtn.disabled = true;
      goalInput.disabled = true;
      appendLine('Starting...');
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => {
        ws.send(JSON.stringify({goal: goalInput.value}));
      };
      ws.onmessage = (ev) => appendLine(ev.data);
      ws.onerror = () => appendLine('WebSocket error.');
      ws.onclose = () => {
        appendLine('Closed.');
        ws = null;
        startBtn.disabled = false;
        goalInput.disabled = false;
      };
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
        orchestrator = Orchestrator(goal=goal, console=ws_console)
        await orchestrator.run()
        await ws_console.wait()
        await websocket.send_text("Furrow run finished.")
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(f"Error: {exc}")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
