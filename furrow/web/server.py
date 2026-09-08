from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class WebSocketConsole:
    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    def print(self, *args, **kwargs) -> None:
        try:
            message = "".join(str(a) for a in args)
            self._websocket.send_text(message)
        except Exception:
            pass


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
      out.textContent = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        out.textContent += ev.data + '\\n';
      };
      ws.onclose = () => {
        out.textContent += '\\nClosed.\\n';
      };
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
        model = data.get("model")
        
        if model:
            from furrow.config import settings
            settings.model = model
        
        ws_console = WebSocketConsole(websocket)
        
        # Temporarily replace the global console with our WebSocket console
        import furrow.core.orchestrator as orchestrator_module
        original_console = orchestrator_module.console
        orchestrator_module.console = ws_console
        
        try:
            orchestrator = Orchestrator(goal=goal)
            await orchestrator.run()
        finally:
            orchestrator_module.console = original_console
            
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
