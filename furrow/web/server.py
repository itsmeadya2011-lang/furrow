from __future__ import annotations

import asyncio

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

logger = structlog.get_logger(__name__)

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class WebSocketFile:
    """File-like wrapper that forwards writes to a WebSocket as text."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket

    def write(self, data: str) -> int:
        if data:
            asyncio.ensure_future(self._send(data))
        return len(data)

    async def _send(self, data: str) -> None:
        try:
            await self.websocket.send_text(data)
        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa: BLE001 - debug-level fallback logging
            logger.debug("websocket_send_failed", error=str(e))

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
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
"""
    )


@app.get("/health")
async def health() -> dict[str, str]:
    logger.info("health_check")
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    ws_file = WebSocketFile(websocket)
    ws_console = Console(file=ws_file, force_terminal=False, color_system=None)  # type: ignore[arg-type]

    import furrow.core.orchestrator as orch_mod

    original_console = orch_mod.console
    orch_mod.console = ws_console

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        logger.info("websocket_session_started", goal=goal)

        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run()

        logger.info("websocket_session_completed")
        await websocket.send_text("\n[Session complete]")
    except WebSocketDisconnect:
        logger.info("websocket_client_disconnected")
    except Exception as e:  # noqa: BLE001 - intentional catch for client error reporting
        logger.error("websocket_error", error=str(e))
        await websocket.send_text(f"\n[ERROR: {e}]")
    finally:
        orch_mod.console = original_console
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
