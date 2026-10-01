from __future__ import annotations

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

logger = structlog.get_logger(__name__)

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
      const goal = document.getElementById('goal').value;
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onerror = (ev) => out.textContent += '\\nError: ' + ev + '\\n';
      ws.onopen = () => ws.send(JSON.stringify({goal: goal}));
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
        if not goal:
            await websocket.send_text("Error: no goal provided")
            await websocket.close()
            return

        async def handle_output(msg: str) -> None:
            await websocket.send_text(msg)

        orchestrator = Orchestrator(goal=goal, output_handler=handle_output)
        await orchestrator.run()
        await websocket.send_text("DONE")
    except WebSocketDisconnect:
        logger.info("websocket_disconnect")
    except Exception as e:
        logger.exception("websocket_error", error=str(e))
        await websocket.send_text(f"Error: {e}")
    finally:
        # Best-effort close; the socket may already be closed.
        try:
            await websocket.close()
        except WebSocketDisconnect:
            pass


@app.post("/start")
async def start_http(request: StartRequest) -> dict[str, str]:
    """Start an orchestration run over HTTP (non-streaming).

    Returns a summary once the run completes.  For real-time streaming,
    use the WebSocket endpoint at ``/ws`` instead.
    """
    if request.model:
        settings.model = request.model
    orchestrator = Orchestrator(goal=request.goal)
    await orchestrator.run()
    return {"status": "complete" if orchestrator._done else "stopped"}


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
