from __future__ import annotations

import asyncio
import io
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None
    provider: Optional[str] = None
    max_cycles: Optional[int] = None


class WebSocketStream(io.StringIO):
    """A StringIO that forwards written text to a WebSocket queue."""

    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self.websocket = websocket
        self.loop = loop

    def write(self, text: str) -> int:
        if text.strip():
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_text(text), self.loop
            )
        return len(text)

    def flush(self) -> None:
        pass


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <div>
    <input id="goal" placeholder="Enter goal" required />
    <button onclick="start()">Start</button>
    <button onclick="stop()">Stop</button>
  </div>
  <pre id="out"></pre>
  <script>
    let ws = null;
    const out = document.getElementById('out');
    function start() {
      out.textContent += '\\nStarting...\\n';
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => { out.textContent += ev.data + '\\n'; out.scrollTop = out.scrollHeight; };
      ws.onclose = () => { out.textContent += '\\nClosed.\\n'; ws = null; };
      ws.onerror = (ev) => { out.textContent += '\\nError.\\n'; };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    }
    function stop() {
      if (ws) ws.close();
    }
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()
    stream = WebSocketStream(websocket, loop)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        provider = data.get("provider")
        max_cycles = data.get("max_cycles")

        orchestrator = Orchestrator(goal=goal)
        if model:
            orchestrator.client.settings.model = model
        if provider:
            from furrow.config import Provider
            orchestrator.client.settings.provider = Provider(provider)
        if max_cycles is not None:
            orchestrator.settings.max_cycles = max_cycles

        # Run the orchestrator while streaming output to the WebSocket.
        await orchestrator.run()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
