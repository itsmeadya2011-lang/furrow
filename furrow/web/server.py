from __future__ import annotations

import json

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class EventBroadcaster:
    """Broadcasts events to all connected websockets."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def broadcast(self, event: dict[str, any]) -> None:
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(json.dumps(event, default=str))
            except Exception:  # noqa: BLE001 - drop dead websockets on any send failure
                dead.append(ws)
        for ws in dead:
            if ws in self._connections:
                self._connections.remove(ws)

    def subscribe(self, ws: WebSocket) -> None:
        self._connections.append(ws)

    def unsubscribe(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)


_broadcaster = EventBroadcaster()


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
      ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        out.textContent += JSON.stringify(data, null, 2) + '\\n---\\n';
      };
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
    _broadcaster.subscribe(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        settings = Settings()
        if model:
            settings.model = model

        orchestrator = Orchestrator(goal=goal, settings=settings)

        async def on_event(event: dict[str, any]) -> None:
            await _broadcaster.broadcast(event)

        orchestrator.on_event(on_event)

        await orchestrator.run()
        await _broadcaster.broadcast({"type": "done"})
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001 - broadcast unexpected errors to the client
        await _broadcaster.broadcast({"type": "error", "error": str(e)})
    finally:
        _broadcaster.unsubscribe(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)