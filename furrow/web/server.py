from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active:
            self.active.remove(websocket)

    async def send_personal_message(self, websocket: WebSocket, text: str) -> None:
        await websocket.send_text(text)

    async def broadcast(self, text: str) -> None:
        for websocket in list(self.active):
            await websocket.send_text(text)


manager = ConnectionManager()


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


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
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        async def on_message(text: str) -> None:
            await manager.send_personal_message(websocket, text)

        orchestrator = Orchestrator(goal=goal, output_callback=on_message)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await manager.send_personal_message(
            websocket, f"[ERROR] {type(exc).__name__}: {exc}"
        )
    finally:
        manager.disconnect(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
