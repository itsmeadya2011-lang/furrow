from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator
from furrow.logging import get_logger

app = FastAPI(title="Furrow")

log = get_logger(__name__)


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

        async def send_output(message: str) -> None:
            try:
                # Strip rich markup tags for plain text display
                import re
                clean = re.sub(r"\[/?[a-zA-Z0-9_]+\]", "", message)
                await websocket.send_text(clean)
            except Exception:
                pass

        orchestrator = Orchestrator(
            goal=goal,
            output_callback=send_output,
        )
        await orchestrator.run()
        await websocket.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error("WebSocket error", error=str(e))
        try:
            await websocket.send_text(f"\n[error]Server error: {e}[/error]")
        except Exception:
            pass
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    log.info("Starting Furrow web server", host=host, port=port)
    uvicorn.run(app, host=host, port=port)
