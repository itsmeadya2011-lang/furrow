from __future__ import annotations

import asyncio
from io import StringIO

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


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; margin: 2rem; }
  h1 { color: #2e7d32; }
  #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px;
         height: 80vh; overflow-y: scroll; white-space: pre-wrap; }
  input { width: 80%; padding: 0.5rem; }
  button { padding: 0.5rem 1rem; }
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
      out.textContent = '';
      out.textContent += 'Starting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => { out.textContent += ev.data + '\\n'; out.scrollTop = out.scrollHeight; };
      ws.onclose = () => out.textContent += '\\n[connection closed]\\n';
      ws.onerror = (ev) => out.textContent += '\\n[error]\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""
    )


class WebSocketConsole:
    """A rich.Console whose recorded output is streamed to a WebSocket client.

    The :meth:`periodic_flush` background task can be run alongside the
    orchestrator so the browser receives near-real-time updates instead of a
    single blob at the end.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._console = Console(
            force_terminal=False,
            width=120,
            record=True,
            # Discard raw stdout – we push through send_text instead.
            file=StringIO(),
        )
        self._closed = False
        self._send_lock = asyncio.Lock()

    @property
    def console(self) -> Console:
        return self._console

    async def flush(self) -> None:
        """Send everything printed since the last flush to the client."""
        if self._closed:
            return
        output = self._console.export_text(clear=True)
        if output:
            async with self._send_lock:
                try:
                    await self._websocket.send_text(output)
                except Exception:  # noqa: BLE001 — connection may drop
                    # Client disconnected while we were trying to send.
                    self._closed = True

    async def periodic_flush(self, interval: float = 0.5) -> None:
        """Continuously flush recorded output until ``close`` is called."""
        while not self._closed:
            await self.flush()
            await asyncio.sleep(interval)

    async def close(self) -> None:
        await self.flush()
        self._closed = True


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    ws_console: WebSocketConsole | None = None
    flush_task: asyncio.Task | None = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        ws_console = WebSocketConsole(websocket)

        settings = Settings()
        if model:
            settings.model = model
            settings.worker_model = model
            settings.planner_model = model
            settings.tester_model = model

        orchestrator = Orchestrator(
            goal=goal, custom_console=ws_console.console
        )
        if model:
            orchestrator.client.settings = settings

        # Stream output to the browser while the loop runs.
        flush_task = asyncio.create_task(ws_console.periodic_flush())

        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 — top-level guard for unexpected errors
        try:
            await websocket.send_text(f"\n[error] {exc}\n")
        except Exception:  # noqa: S110, BLE001 — best-effort error notification
            pass
    finally:
        if flush_task is not None:
            flush_task.cancel()
        if ws_console is not None:
            await ws_console.close()
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
