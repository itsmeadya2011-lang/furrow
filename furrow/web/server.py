from __future__ import annotations

import asyncio
from io import StringIO

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


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
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
    """
    )


class StreamConsole:
    """A rich Console that forwards rendered output to a WebSocket.

    Captures output written to stdout and streams it to the client so the
    web UI can show live progress during the development loop.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._buffer = StringIO()
        self._websocket = websocket
        self._console = Console(file=self._buffer, force_terminal=False)

    @property
    def console(self) -> Console:
        return self._console

    async def flush(self) -> None:
        """Send any buffered content to the WebSocket."""
        content = self._buffer.getvalue()
        if content:
            await self._websocket.send_text(content)
            self._buffer.seek(0)
            self._buffer.truncate(0)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    stream = StreamConsole(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        # Create a background task for the orchestrator
        async def run_orchestrator() -> None:
            orch = Orchestrator(goal=goal)
            # Monkey-patch the module-level console to stream output
            import furrow.core.orchestrator as orch_mod

            original_console = orch_mod.console
            orch_mod.console = stream.console
            try:
                # Periodically flush the stream console
                flush_task = asyncio.create_task(_periodic_flush(stream, websocket))
                await orch.run()
                flush_task.cancel()
                await stream.flush()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                await websocket.send_text(f"\n[red]{e}[/red]\n")
            finally:
                orch_mod.console = original_console

        await run_orchestrator()
        await websocket.close()
    except WebSocketDisconnect:
        pass


async def _periodic_flush(stream: StreamConsole, websocket: WebSocket) -> None:
    """Periodically flush console output to the WebSocket client."""
    while True:
        await asyncio.sleep(0.5)
        try:
            await stream.flush()
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001
            return


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
