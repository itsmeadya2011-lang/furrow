from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


async def _safe_send(websocket: WebSocket, message: str) -> None:
    """Send ``message`` as plain text, ignoring a client that already went away."""
    try:
        await websocket.send_text(message)
    except Exception as exc:
        # A disconnected client must never break the orchestrator loop. Any
        # failure here (WebSocketDisconnect, RuntimeError, OSError, ...) just
        # means there is nobody left to talk to.
        logger.debug("websocket send failed: %s", exc, exc_info=True)


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
        try:
            data = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        except Exception as exc:
            # Non-JSON text, a binary frame, or a payload that is not an object.
            logger.warning("could not read websocket request: %s", exc, exc_info=True)
            await _safe_send(websocket, f"Error: could not read request ({exc}).")
            return

        if not isinstance(data, dict):
            await _safe_send(websocket, "Error: request must be a JSON object.")
            return

        try:
            request = StartRequest(**data)
        except ValidationError as exc:
            await _safe_send(websocket, f"Error: {exc}")
            return

        goal = request.goal.strip()
        if not goal:
            await _safe_send(websocket, "Error: 'goal' must not be empty.")
            return

        await _safe_send(websocket, f"Goal received: {request.goal}")
        try:
            orchestrator = Orchestrator(goal=request.goal)
            await orchestrator.run()
        except (ValueError, RuntimeError) as exc:
            await _safe_send(websocket, f"Error: {exc}")
        except Exception as exc:
            logger.exception("orchestrator loop failed")
            await _safe_send(websocket, f"Error: {type(exc).__name__}: {exc}")
        else:
            await _safe_send(websocket, f"Loop finished after {orchestrator.cycles} cycle(s).")
    except WebSocketDisconnect:
        pass
    finally:
        await _safe_send(websocket, "Done.")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
