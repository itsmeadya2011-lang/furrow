from __future__ import annotations

import asyncio
import sys
from io import StringIO

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


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
  <pre id="out" style="background:#1e1e1e; color:#d4d4d4; padding:1rem; max-height:60vh; overflow:auto;"></pre>
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


class WebSocketStream:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.buffer = StringIO()

    def write(self, text: str) -> int:
        self.buffer.write(text)
        # Send complete lines
        value = self.buffer.getvalue()
        lines = value.split('\n')
        # Keep the last incomplete line in buffer
        self.buffer = StringIO()
        if lines[-1]:
            self.buffer.write(lines[-1])
        for line in lines[:-1]:
            if line:
                asyncio.create_task(self.websocket.send_text(line))
        return len(text)

    def flush(self) -> None:
        value = self.buffer.getvalue()
        if value:
            asyncio.create_task(self.websocket.send_text(value))
            self.buffer = StringIO()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        stream = WebSocketStream(websocket)
        
        # Capture stdout/stderr and send to websocket
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stream
        sys.stderr = stream
        
        try:
            orchestrator = Orchestrator(goal=goal)
            if model:
                orchestrator.settings.model = model
            await orchestrator.run()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            stream.flush()
    except WebSocketDisconnect:
        pass
    except OSError as e:
        await websocket.send_text(f"Error: {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
