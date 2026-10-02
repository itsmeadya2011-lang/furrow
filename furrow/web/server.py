from __future__ import annotations

import asyncio
import io
from typing import Optional

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


# --- WebSocket connection management ---


class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)


manager = ConnectionManager()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=_INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    # Use a StringIO buffer; Rich writes synchronously to it, a background
    # task drains it to the WebSocket periodically.
    buffer = io.StringIO(newline="\n")
    ws_console = Console(
        file=buffer,
        force_terminal=False,
        no_color=True,
        width=120,
        highlight=False,
        soft_wrap=True,
    )

    async def flush_loop() -> None:
        """Periodically flush buffered console output to the WebSocket."""
        while True:
            content = buffer.getvalue()
            if content:
                # Reset buffer
                buffer.seek(0)
                buffer.truncate(0)
                await websocket.send_text(content)
            await asyncio.sleep(0.3)

    flush_task = asyncio.create_task(flush_loop())

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        settings = Settings()
        if model:
            settings.model = model
            settings.worker_model = model
            settings.tester_model = model

        orchestrator = Orchestrator(
            goal=goal,
            settings=settings,
            rich_console=ws_console,
        )
        await orchestrator.run()

        # Final flush
        remaining = buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
        if remaining:
            await websocket.send_text(remaining)
        await websocket.send_text("\n[DONE] Goal complete.")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"\n[ERROR] {type(e).__name__}: {e}")
        except Exception:
            pass
    finally:
        flush_task.cancel()
        try:
            await flush_task
        except asyncio.CancelledError:
            pass
        manager.disconnect(websocket)


_INDEX_HTML = """
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
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
