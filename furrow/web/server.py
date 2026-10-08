from __future__ import annotations

import asyncio
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

        import io
        from rich.console import Console

        # Create a queue for streaming output
        output_queue: asyncio.Queue[str] = asyncio.Queue()

        # Create a console that writes to the queue
        class WebSocketConsole(Console):
            def __init__(self, queue: asyncio.Queue[str], *args, **kwargs):
                super().__init__(*args, **kwargs, force_terminal=False, file=io.StringIO())
                self._queue = queue

            def print(self, *args, **kwargs):
                # Capture what would be printed and put it in the queue
                with self.capture() as capture:
                    super().print(*args, **kwargs)
                text = capture.get()
                self._queue.put_nowait(text)

        ws_console = WebSocketConsole(output_queue)

        # Patch the orchestrator's console
        from furrow.core import orchestrator as orch_module
        original_console = orch_module.console
        orch_module.console = ws_console

        orchestrator = Orchestrator(goal=goal)
        run_task = asyncio.create_task(orchestrator.run())

        # Stream output to client
        try:
            while True:
                try:
                    # Wait for output with a short timeout
                    line = await asyncio.wait_for(output_queue.get(), timeout=0.5)
                    await websocket.send_text(line)
                except asyncio.TimeoutError:
                    if run_task.done():
                        break
                    # Send a heartbeat to keep connection alive
                    try:
                        await websocket.send_text(".")
                    except Exception:
                        return
        finally:
            orch_module.console = original_console

        # Drain remaining queue
        while not output_queue.empty():
            line = await output_queue.get()
            await websocket.send_text(line)

        # Send final result
        try:
            await run_task  # Get any exception
            await websocket.send_text("\n[Furrow] Goal processing complete.")
        except Exception as e:
            await websocket.send_text(f"\n[Furrow] Error: {e}")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
