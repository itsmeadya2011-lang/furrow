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
        
        # Custom console that sends output to websocket
        class WebSocketConsole:
            def __init__(self, ws: WebSocket):
                self.ws = ws
            
            def print(self, *args, **kwargs):
                msg = " ".join(str(a) for a in args)
                asyncio.create_task(self.ws.send_text(msg))
        
        ws_console = WebSocketConsole(websocket)
        
        # Create orchestrator with custom console
        orchestrator = Orchestrator(goal=goal)
        # Monkey-patch the console for this orchestrator run
        import furrow.core.orchestrator as orch_module
        original_console = orch_module.console
        orch_module.console = ws_console
        
        try:
            await orchestrator.run()
        finally:
            orch_module.console = original_console
            
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        await websocket.send_text(f"Error: {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
