from __future__ import annotations

import asyncio
import re
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
import furrow.core.orchestrator as orchestrator_module

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketWriter:
    """A file-like object that writes to a WebSocket."""
    
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self._buffer = ""
    
    def write(self, data: str) -> int:
        # Buffer partial lines and send complete lines
        self._buffer += data
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            # Strip ANSI escape codes
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
            asyncio.create_task(self.websocket.send_text(clean + '\n'))
        return len(data)
    
    def flush(self) -> None:
        if self._buffer:
            clean = re.sub(r'\x1b\[[0-9;]*m', '', self._buffer)
            asyncio.create_task(self.websocket.send_text(clean))
            self._buffer = ""


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    writer = WebSocketWriter(websocket)
    
    # Create a console that writes to the WebSocket
    ws_console = Console(file=writer, force_terminal=False, width=120)
    
    # Save original console and replace it
    original_console = orchestrator_module.console
    orchestrator_module.console = ws_console
    
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run()
        await websocket.send_text("\n=== Furrow completed ===\n")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"\nError: {e}\n")
        except Exception:
            pass
    finally:
        # Restore original console
        orchestrator_module.console = original_console
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
