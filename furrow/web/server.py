from __future__ import annotations

import asyncio
import enum
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Active WebSocket sessions keyed by connection id. Only one goal may run at a
# time per process; additional connections are rejected with a status message.
_active_sessions: dict[int, asyncio.Task] = {}


class ConnectionState(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketConsoleFile:
    """File-like object that buffers writes and flushes them to a WebSocket.

    Compatible with Rich's Console (which calls write() and flush()). The flush
    is async-safe: it schedules a coroutine on the running event loop to send
    the buffered text to the client.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        return len(text)

    def flush(self) -> None:
        if not self._buf:
            return
        buf = self._buf
        self._buf = ""
        loop = asyncio.get_running_loop()
        loop.create_task(self.websocket.send_text(buf))

    def isatty(self) -> bool:
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


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    # Reject new connections while a goal is already running.
    if _active_sessions:
        await websocket.send_text(
            "A goal is already running. Please wait for it to finish."
        )
        await websocket.close()
        return

    connection_id = id(websocket)
    state = ConnectionState.IDLE

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal:
            await websocket.send_text("Error: no goal provided.")
            await websocket.close()
            return

        state = ConnectionState.RUNNING
        _active_sessions[connection_id] = asyncio.current_task()

        ws_console = Console(
            file=WebSocketConsoleFile(websocket),
            force_terminal=False,
            width=120,
        )
        orchestrator = Orchestrator(goal=goal, console=ws_console)
        await orchestrator.run()

        await websocket.send_text("\n[bold green]Done.[/bold green]")
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(f"[bold red]Error:[/bold red] {exc}")
        except Exception:
            pass
    finally:
        _active_sessions.pop(connection_id, None)
        state = ConnectionState.IDLE
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)