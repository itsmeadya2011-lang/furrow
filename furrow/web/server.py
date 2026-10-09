from __future__ import annotations

import asyncio
import io
import logging

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class StreamingConsole:
    """Console that streams output to a WebSocket."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.buffer = ""

    def print(self, *args, **kwargs):
        # Convert rich output to plain text
        string_io = io.StringIO()
        temp_console = Console(file=string_io, force_terminal=False, width=120)
        temp_console.print(*args, **kwargs)
        output = string_io.getvalue()

        if output.strip():
            asyncio.create_task(self._send(output))

    async def _send(self, message: str):
        try:
            await self.websocket.send_text(message)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to send websocket message: %s", e)



@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
        h1 { color: #4ec9b0; }
        #form { margin-bottom: 20px; }
        #goal { width: 70%; padding: 10px; background: #2d2d2d; border: 1px solid #3c3c3c; color: #d4d4d4; }
        button { padding: 10px 20px; background: #4ec9b0; border: none; color: #1e1e1e; cursor: pointer; font-weight: bold; }
        button:hover { background: #3db8a0; }
        #out { background: #1e1e1e; border: 1px solid #3c3c3c; padding: 15px; height: 500px; overflow-y: auto; white-space: pre-wrap; font-size: 13px; line-height: 1.5; }
        .cycle { color: #569cd6; font-weight: bold; }
        .success { color: #4ec9b0; }
        .error { color: #f44747; }
        .warning { color: #dcdcaa; }
        .info { color: #9cdcfe; }
    </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      out.textContent += '\\nStarting: ' + goal + '\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const text = ev.data;
        // Color code output
        let colored = text
          .replace(/═══ Cycle (\\d+) ═══/g, '<span class="cycle">═══ Cycle $1 ═══</span>')
          .replace(/\\[green\\](.*?)\\[\\/green\\]/g, '<span class="success">$1</span>')
          .replace(/\\[red\\](.*?)\\[\\/red\\]/g, '<span class="error">$1</span>')
          .replace(/\\[yellow\\](.*?)\\[\\/yellow\\]/g, '<span class="warning">$1</span>')
          .replace(/\\[cyan\\](.*?)\\[\\/cyan\\]/g, '<span class="info">$1</span>')
          .replace(/\\[bold (\\w+)\\]/g, '<span class="$1">')
          .replace(/\\[\\/bold\\]/g, '</span>');
        out.innerHTML += colored + '\\n';
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.innerHTML += '\\n<span class="warning">Connection closed.</span>\\n';
      ws.send(JSON.stringify({goal: goal}));
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

        # Create orchestrator with streaming console
        streaming_console = StreamingConsole(websocket)

        # Monkey-patch console for this orchestrator
        import furrow.core.orchestrator as orch_module
        original_console = orch_module.console
        orch_module.console = streaming_console

        try:
            orchestrator = Orchestrator(goal=goal)
            await orchestrator.run()
        finally:
            orch_module.console = original_console

    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log = logging.getLogger(__name__)
        log.warning("WebSocket error: %s", e)
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception as inner_e:  # noqa: BLE001
            log.warning("Failed to send error message: %s", inner_e)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)