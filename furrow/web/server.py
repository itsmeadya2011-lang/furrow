from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")

logger = logging.getLogger("furrow.web")


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

        settings = Settings()
        if model:
            settings.model = model

        client = LLMClient(settings=settings)

        async def send_message(msg: str) -> None:
            await websocket.send_text(msg)

        orchestrator = Orchestrator(goal=goal, client=client, on_message=send_message)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("Orchestrator error in WebSocket session")
        try:
            await websocket.send_text(f"[error] {e}")
        except WebSocketDisconnect:
            pass
    finally:
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
