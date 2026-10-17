from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

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
        orchestrator = Orchestrator(goal=goal)

        # Stream output to websocket
        original_console = orch_module.console
        import io

        class WebSocketConsole(Console):
            def __init__(self, ws: WebSocket):
                super().__init__(file=io.StringIO(), width=120)
                self.ws = ws

            def _write_buffer(self):
                buf = self.file.getvalue()
                if buf:
                    self.file.truncate(0)
                    self.file.seek(0)
                    try:
                        asyncio.create_task(self.ws.send_text(buf))
                    except Exception:
                        pass

        ws_console = WebSocketConsole(websocket)
        orch_module.console = ws_console
        try:
            await orchestrator.run()
        finally:
            await websocket.send_text("\n[COMPLETE]")
    except WebSocketDisconnect:
        pass
    finally:
        try:
            orch_module.console = original_console
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
