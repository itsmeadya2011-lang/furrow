from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient
from furrow.logging import get_logger

logger = get_logger("server")

app = FastAPI(title="Furrow")

# Global orchestrator instance for shared state
_orchestrator: Orchestrator | None = None
_orchestrator_lock = asyncio.Lock()


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


async def get_orchestrator(goal: str, model: str | None = None, websocket: WebSocket | None = None) -> Orchestrator:
    global _orchestrator
    async with _orchestrator_lock:
        if _orchestrator is None:
            client = LLMClient()
            if model:
                client.settings.model = model
            output_callback = None
            if websocket:
                async def send_output(message: str):
                    try:
                        await websocket.send_text(message)
                    except Exception as e:  # noqa: BLE001
                        logger.debug("websocket_send_failed", error=str(e))
                output_callback = send_output
            _orchestrator = Orchestrator(goal=goal, client=client, output_callback=output_callback)
        elif _orchestrator.goal != goal:
            # New goal, create new orchestrator
            client = LLMClient()
            if model:
                client.settings.model = model
            output_callback = None
            if websocket:
                async def send_output(message: str):
                    try:
                        await websocket.send_text(message)
                    except Exception as e:  # noqa: BLE001
                        logger.debug("websocket_send_failed", error=str(e))
                output_callback = send_output
            _orchestrator = Orchestrator(goal=goal, client=client, output_callback=output_callback)
        elif websocket and _orchestrator._output_callback is None:
            # Update callback for existing orchestrator
            async def send_output(message: str):
                try:
                    await websocket.send_text(message)
                except Exception as e:  # noqa: BLE001
                    logger.debug("websocket_send_failed", error=str(e))
            _orchestrator._output_callback = send_output
        return _orchestrator


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
        
        orchestrator = await get_orchestrator(goal, model, websocket)
        
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        await websocket.send_text(f"Error: {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
