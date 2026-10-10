from __future__ import annotations

import asyncio

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
        orchestrator = Orchestrator(goal=goal)

        # Capture Rich console output and forward to WebSocket
        from io import StringIO

        from rich.console import Console

        buffer = StringIO()
        ws_console = Console(file=buffer, force_terminal=False, width=120)

        async def send_output() -> None:
            while True:
                await asyncio.sleep(0.1)
                content = buffer.getvalue()
                if content:
                    buffer.seek(0)
                    buffer.truncate(0)
                    try:
                        await websocket.send_text(content)
                    except WebSocketDisconnect:
                        break

        send_task = asyncio.create_task(send_output())

        # Temporarily swap the global console to our buffer-backed one
        import furrow.core.orchestrator as orch_module
        original_console = orch_module.console
        orch_module.console = ws_console

        try:
            await orchestrator.run()
        finally:
            orch_module.console = original_console
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass

        # Flush remaining buffer
        remaining = buffer.getvalue()
        if remaining:
            try:
                await websocket.send_text(remaining)
            except WebSocketDisconnect:
                pass

        try:
            await websocket.send_text("\n[Done]\n")
        except WebSocketDisconnect:
            pass
    finally:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)