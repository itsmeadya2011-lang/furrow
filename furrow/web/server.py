from __future__ import annotations

import re
import traceback

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    provider: str | None = None


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
      ws.onerror = (ev) => out.textContent += '\\nError: ' + ev.message + '\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


# Regex to strip rich markup tags ([bold], [/], etc.) for plain-text WebSocket clients
_RICH_TAG_RE = re.compile(r"\[/?[a-zA-Z_][^\]]*\]")


async def _ws_message_handler(websocket: WebSocket):
    """Return an output handler that sends messages over the WebSocket."""

    async def _send(message: str) -> None:
        clean = _RICH_TAG_RE.sub("", message)
        await websocket.send_text(clean)

    return _send


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        # Build settings with optional overrides from the request
        ws_settings = Settings()
        if data.get("model"):
            ws_settings.model = data["model"]
            ws_settings.planner_model = data["model"]
            ws_settings.worker_model = data["model"]
            ws_settings.tester_model = data["model"]
        if data.get("provider"):
            ws_settings.provider = Provider(data["provider"])

        output_handler = await _ws_message_handler(websocket)
        orchestrator = Orchestrator(
            goal=goal, settings=ws_settings, output_handler=output_handler
        )
        await orchestrator.run()
        await websocket.send_text("--- Furrow finished ---")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"ERROR: {e}")
        traceback.print_exc()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
