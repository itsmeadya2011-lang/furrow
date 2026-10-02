from __future__ import annotations

import asyncio
import json

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator
from furrow.logging_setup import get_logger

__all__ = ["StartRequest", "app", "run"]

app = FastAPI(title="Furrow")
logger = get_logger("server")


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
    <button type="button" id="stopBtn" style="display:none;">Stop</button>
  </form>
  <div id="progress"></div>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const progressEl = document.getElementById('progress');
    const stopBtn = document.getElementById('stopBtn');
    let ws;
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      ws = new WebSocket('ws://' + location.host + '/ws');
      stopBtn.style.display = 'inline-block';
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === 'progress') {
            progressEl.textContent += (data.message || '') + '\\n';
          } else if (data.type === 'error') {
            out.textContent += '[ERROR] ' + (data.message || '') + '\\n';
          } else {
            out.textContent += ev.data + '\\n';
          }
        } catch {
          out.textContent += ev.data + '\\n';
        }
      };
      ws.onclose = () => {
        out.textContent += '\\nClosed.\\n';
        stopBtn.style.display = 'none';
      };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
    stopBtn.onclick = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({action: 'stop'}));
      }
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("websocket_connect")
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        logger.info("goal_start", goal=goal)
        orchestrator = Orchestrator(goal=goal)

        async def send_progress(msg: str) -> None:
            try:
                await websocket.send_text(msg)
            except (WebSocketDisconnect, RuntimeError):
                pass

        orchestrator.progress_callback = (
            lambda msg: asyncio.create_task(send_progress(msg))
        )

        async def run_orchestrator() -> None:
            await orchestrator.run()

        async def listen_for_stop() -> None:
            try:
                while True:
                    msg = await websocket.receive_text()
                    msg_data = json.loads(msg)
                    if msg_data.get("action") == "stop":
                        logger.info("stop_request")
                        orchestrator.stop()
            except WebSocketDisconnect:
                pass

        _, pending = await asyncio.wait(
            {run_orchestrator(), listen_for_stop()},
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Cancel any remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    except WebSocketDisconnect:
        logger.info("websocket_disconnect")
    except Exception as e:  # noqa: BLE001
        logger.error("websocket_error", error=str(e))
        await websocket.send_text(
            json.dumps({"type": "error", "message": str(e)})
        )
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
