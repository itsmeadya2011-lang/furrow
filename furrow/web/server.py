from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console
from rich.redirect_stdout import redirect_stdout

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


class WebSocketStream:
    """A writable stream that forwards everything to a WebSocket client."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._buffer: list[str] = []

    async def send(self, data: Any) -> None:
        await self._websocket.send_text(str(data))

    async def flush(self) -> None:
        if self._buffer:
            await self.send("".join(self._buffer))
            self._buffer.clear()

    async def write(self, data: Any) -> None:
        self._buffer.append(str(data))
        if "\n" in str(data):
            await self.flush()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)

        # Redirect console output so it streams to the WebSocket client.
        stream = WebSocketStream(websocket)
        console = Console(
            file=stream,  # type: ignore[arg-type]
            force_terminal=False,
            no_color=False,
            width=120,
            highlight=False,
        )

        # Patch the orchestrator's module-level console for live streaming.
        import furrow.core.orchestrator as orch_mod

        original_console = orch_mod.console
        orch_mod.console = console

        try:
            with redirect_stdout(stream):
                await orchestrator.run()
        finally:
            orch_mod.console = original_console
            await stream.flush()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"Error: {e}")
    finally:
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
