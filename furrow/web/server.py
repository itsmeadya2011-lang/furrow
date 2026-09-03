from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow import core
from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active_connections.append(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, msg: str) -> None:
        dead: list[WebSocket] = []
        for ws in list(self.active_connections):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


class StreamingConsole(Console):
    """Console that mirrors rendered output to a ConnectionManager.

    Used to monkeypatch the orchestrator's module-level console so rich prints
    are also pushed to connected websocket clients without touching orchestrator.
    """

    def __init__(
        self,
        manager: ConnectionManager,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._manager = manager
        self._loop = loop

    def _emit(self, text: str) -> None:
        if not text:
            return
        # Schedule broadcast on the running event loop; orchestrator runs in the
        # same loop as FastAPI, so ensure_future is safe here.
        try:
            asyncio.ensure_future(
                self._manager.broadcast(text), loop=self._loop
            )
        except RuntimeError:
            pass

    def print(self, *objects: object, **kwargs: object) -> None:
        capture = Console(
            record=True,
            no_color=True,
            force_terminal=False,
            width=self.width,
            highlight=False,
            markup=False,
        )
        try:
            capture.print(*objects, **kwargs)
            rendered = capture.export_text(styles=False)
        except Exception:
            rendered = " ".join(str(o) for o in objects)
        if rendered:
            self._emit(rendered)
        super().print(*objects, **kwargs)


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
    <input id="model" placeholder="model (optional)" />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\nStarting...\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === 'done') { out.textContent += '\n[done]\n'; return; }
          if (data.type === 'log') { out.textContent += data.text + '\n'; return; }
        } catch (_) { out.textContent += ev.data + '\n'; }
      };
      ws.onclose = () => out.textContent += '\nClosed.\n';
      ws.send(JSON.stringify({
        goal: document.getElementById('goal').value,
        model: document.getElementById('model').value || null,
      }));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    manager = ConnectionManager()
    await manager.connect(websocket)
    original_console = core.orchestrator.console
    loop = asyncio.get_running_loop()
    streaming = StreamingConsole(manager=manager, loop=loop)
    core.orchestrator.console = streaming
    try:
        data = await websocket.receive_json()
        goal = str(data.get("goal", ""))
        model = data.get("model")
        if model:
            settings.model = model
        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run()
        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        await websocket.send_json({"type": "log", "text": f"Error: {exc}"})
    finally:
        core.orchestrator.console = original_console
        await manager.disconnect(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)