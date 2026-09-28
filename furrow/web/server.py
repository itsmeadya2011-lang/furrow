from __future__ import annotations

import re

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


_RICH_MARKUP_RE = re.compile(r"\[/?(?:[a-zA-Z0-9_ \-]|#[0-9a-fA-F]{3,8}){1,60}\]")


def _to_plain(msg: str) -> str:
    text = _RICH_MARKUP_RE.sub("", msg)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


INDEX_HTML = """<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <p><a href="/health">health</a></p>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data;
      ws.onclose = () => out.textContent += '\\n[closed]\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        async def ws_output(msg: str) -> None:
            plain = _to_plain(msg) + "\n"
            try:
                await websocket.send_text(plain)
            except Exception:  # noqa: BLE001,S110  # client may have disconnected
                pass

        orchestrator = Orchestrator(goal=goal, output=ws_output)
        await orchestrator.run()
        try:
            await websocket.send_text("\n[closed]\n")
        except Exception:  # noqa: BLE001,S110  # client may have disconnected
            pass
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await websocket.send_text(f"\n[error] {e}\n")
        except Exception:  # noqa: BLE001,S110
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
