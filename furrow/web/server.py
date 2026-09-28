from __future__ import annotations

import contextlib
import re

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

_INDEX_HTML = """<!DOCTYPE html>
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
</html>"""

_RICH_TAG_RE = re.compile(r"\[/?[a-zA-Z][^\]]*\]")


def _strip_rich(text: str) -> str:
    return _RICH_TAG_RE.sub("", text)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=_INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        settings: Settings | None = None
        if model:
            settings = Settings()
            settings.model = model
            settings.planner_model = model
            settings.worker_model = model
            settings.tester_model = model

        async def send(msg: str) -> None:
            with contextlib.suppress(Exception):
                await websocket.send_text(_strip_rich(msg))

        orchestrator = Orchestrator(goal=goal, settings=settings, output_callback=send)
        await orchestrator.run()
        await websocket.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        with contextlib.suppress(Exception):
            await websocket.send_text(f"Error: {e}")
        with contextlib.suppress(Exception):
            await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
