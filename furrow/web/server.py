from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Console, Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketConsole(Console):
    def __init__(self, websocket: WebSocket) -> None:
        super().__init__(force_terminal=False, color_system=None)
        self.websocket = websocket

    def print(self, *args, **kwargs) -> None:
        message = "".join(str(arg) for arg in args)

        async def _send() -> None:
            try:
                await self.websocket.send_json({"type": "log", "message": message})
            except Exception:
                pass

        asyncio.create_task(_send())


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
    <input id="model" placeholder="Model (optional)" />
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
        try {
          const data = JSON.parse(ev.data);
          if (data.type === 'log') {
            out.textContent += data.message + '\\n';
          } else if (data.type === 'done') {
            out.textContent += '\\nDone.\\n';
          }
        } catch {
          out.textContent += ev.data + '\\n';
        }
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      const goal = document.getElementById('goal').value;
      const model = document.getElementById('model').value || undefined;
      ws.send(JSON.stringify({goal, model}));
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
        ws_console = WebSocketConsole(websocket)
        orchestrator = Orchestrator(goal=goal, console=ws_console, model=model)
        await orchestrator.run()
        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
