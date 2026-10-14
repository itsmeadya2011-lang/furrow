from __future__ import annotations

import asyncio
import io

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class StreamingConsole:
    """Console that streams output to a WebSocket."""
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self._buffer = ""

    def print(self, *args, **kwargs):
        # Capture rich output and send to websocket
        buf = io.StringIO()
        temp_console = Console(file=buf, force_terminal=True, width=120)
        temp_console.print(*args, **kwargs)
        output = buf.getvalue()

        # Send to websocket asynchronously
        asyncio.create_task(self._send(output))

    async def _send(self, text: str):
        try:
            await self.websocket.send_text(text)
        except RuntimeError:
            pass  # Connection closed


async def run_orchestrator_with_stream(goal: str, websocket: WebSocket) -> None:
    """Run orchestrator with streaming output to websocket."""
    streaming_console = StreamingConsole(websocket)

    # Monkey-patch the console in orchestrator module
    import furrow.core.orchestrator as orch_module
    original_console = orch_module.console
    orch_module.console = streaming_console  # type: ignore[assignment]

    try:
        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run()
    finally:
        orch_module.console = original_console


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
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        await run_orchestrator_with_stream(goal, websocket)
    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        await websocket.send_text(f"Error: {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
