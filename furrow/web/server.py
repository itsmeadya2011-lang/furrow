from __future__ import annotations

import json

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

logger = structlog.get_logger()

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; 
         min-height: 400px; overflow: auto; white-space: pre-wrap; }
  input { padding: 0.5rem; width: 70%; }
  button { padding: 0.5rem 1rem; }
  .cycle { border-left: 3px solid #4ec9b0; padding-left: 1rem; margin: 1rem 0; }
  .error { color: #f44747; }
  .success { color: #4ec9b0; }
  .warning { color: #cca700; }
  .info { color: #9cdcfe; }
</style>
</head>
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
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'log') {
            out.textContent += msg.content + '\\n';
          } else if (msg.type === 'state') {
            out.textContent += '[State: ' + msg.content + ']\\n';
          }
        } catch {
          out.textContent += ev.data + '\\n';
        }
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
      ws.onerror = (err) => out.textContent += '\\n[Error: ' + err + ']\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


class WebSocketLogger:
    """Wrapper to send logs via WebSocket."""
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def send_text(self, message: str) -> None:
        """Compatibility method for orchestrator."""
        try:
            await self.websocket.send_text(json.dumps({
                "type": "log",
                "content": message,
                "level": "info"
            }))
        except (OSError, RuntimeError) as e:
            logger.debug("websocket_send_error", error=str(e))

    async def send(self, message: str, level: str = "info") -> None:
        try:
            await self.websocket.send_text(json.dumps({
                "type": "log",
                "content": message,
                "level": level
            }))
        except (OSError, RuntimeError) as e:
            logger.debug("websocket_send_error", error=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    ws_logger = WebSocketLogger(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        
        if model:
            from furrow.config import settings
            settings.model = model
        
        orchestrator = Orchestrator(goal=goal, websocket=ws_logger)
        await orchestrator.run()
        await ws_logger.send("Furrow completed.", "success")
    except WebSocketDisconnect:
        pass
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("websocket_error", error=str(e))
        await ws_logger.send(f"Error: {e}", "error")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
