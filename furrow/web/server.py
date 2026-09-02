from __future__ import annotations

import asyncio
import re
from io import StringIO
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
import furrow.core.orchestrator as orch_module

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketConsole(Console):
    """Console that forwards output to a WebSocket."""

    def __init__(self, websocket: WebSocket) -> None:
        super().__init__(file=StringIO(), force_terminal=True)
        self.websocket = websocket

    def print(self, *args, **kwargs) -> None:
        super().print(*args, **kwargs)
        text = " ".join(str(a) for a in args)
        clean = re.sub(r"\[/?[a-z ]+\]", "", text)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send(clean + "\n"))
        except RuntimeError:
            pass

    async def _send(self, text: str) -> None:
        try:
            await self.websocket.send_text(text)
        except Exception:
            pass


class OrchestratorState:
    """Tracks the current orchestrator state for /status endpoint."""

    def __init__(self) -> None:
        self.running = False
        self.goal: Optional[str] = None
        self.cycles = 0
        self.orchestrator: Optional[Orchestrator] = None

    def reset(self) -> None:
        self.running = False
        self.goal = None
        self.cycles = 0
        self.orchestrator = None


state = OrchestratorState()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""<!DOCTYPE html>
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
</html>""")


@app.get("/status")
async def status() -> dict:
    return {
        "running": state.running,
        "goal": state.goal,
        "cycles": state.cycles,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    if state.running:
        await websocket.send_text("An orchestrator is already running. Try again later.")
        await websocket.close()
        return

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        ws_console = WebSocketConsole(websocket)

        state.reset()
        state.running = True
        state.goal = goal

        orchestrator = Orchestrator(goal=goal)
        state.orchestrator = orchestrator

        original_console = orch_module.console
        orch_module.console = ws_console

        task = asyncio.create_task(orchestrator.run())

        async def monitor_disconnect() -> None:
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                task.cancel()

        monitor_task = asyncio.create_task(monitor_disconnect())

        try:
            await task
        except asyncio.CancelledError:
            await websocket.send_text("\n[Cancelled by user]")
        except Exception as e:
            await websocket.send_text(f"\n[Error: {e}]")
        finally:
            monitor_task.cancel()
            state.cycles = orchestrator.cycles
            state.running = False
            orch_module.console = original_console
            await websocket.send_text("\n[Done]")

    except WebSocketDisconnect:
        state.running = False
    except Exception as e:
        print(f"WebSocket error: {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
