from __future__ import annotations

import logging
from contextlib import suppress

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

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
        model = data.get("model")

        custom_settings: Settings | None = None
        if model:
            custom_settings = Settings()
            custom_settings.model = model

        async def send_to_client(message: str) -> None:
            await websocket.send_text(message)

        orchestrator = Orchestrator(
            goal=goal,
            settings=custom_settings,
            output_sink=send_to_client,
        )
        await orchestrator.run()
        await websocket.send_text("\n[Connection closing...]")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("websocket_error")
        try:
            await websocket.send_text(f"\n[Error: {e}]")
        except Exception:
            logger.debug("failed_to_send_error_response", exc_info=True)
    finally:
        with suppress(Exception):
            await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
