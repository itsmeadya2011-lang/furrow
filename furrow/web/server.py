from __future__ import annotations

import asyncio
import json
import re

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging import configure_logging

app = FastAPI(title="Furrow")

_RICH_TAG_RE = re.compile(r"\[/?[a-zA-Z][^\]]*\]")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


HTML_CONTENT = """<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: 'Courier New', monospace; margin: 0; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
    h1 { color: #4CAF50; }
    input[type="text"] { width: 100%; padding: 10px; border: 1px solid #444; border-radius: 4px; background: #2d2d2d; color: #d4d4d4; }
    button { padding: 10px 20px; margin-top: 8px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
    button:hover { background: #45a049; }
    #out { margin-top: 15px; white-space: pre-wrap; background: #121212; padding: 15px; border-radius: 5px; border: 1px solid #333; max-height: 80vh; overflow-y: auto; font-size: 13px; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter your goal for Furrow..." required />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          out.textContent += data.data + '\\n';
        } catch {
          out.textContent += ev.data + '\\n';
        }
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => { out.textContent += '\\n[connection closed]\\n'; out.scrollTop = out.scrollHeight; };
      ws.onerror = (err) => { out.textContent += '\\n[error: ' + err + ']\\n'; };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=HTML_CONTENT)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        settings = Settings()
        configure_logging(settings.log_level)

        def send_output(msg: str) -> None:
            clean = _RICH_TAG_RE.sub("", msg)
            async def _send() -> None:
                try:
                    await websocket.send_text(
                        json.dumps({"type": "output", "data": clean})
                    )
                except Exception:
                    pass  # Client disconnected

            asyncio.create_task(_send())

        orchestrator = Orchestrator(
            goal=goal, settings=settings, on_output=send_output
        )
        await orchestrator.run()
        await websocket.send_text(
            json.dumps({"type": "done", "data": "Furrow completed"})
        )
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "data": str(e)})
            )
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
