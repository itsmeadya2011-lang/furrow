from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import configure_logging, settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


def _strip_rich_markup(text: str) -> str:
    """Remove Rich markup tags from a string."""
    import re
    return re.sub(r"\[/?[a-zA-Z0-9 _#]*?\]", "", text)


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
    let ws = null;

    function appendMessage(type, data) {
      const prefix = type === 'error' ? '[ERROR] ' : '';
      out.textContent += prefix + data + '\\n';
      out.scrollTop = out.scrollHeight;
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      if (ws) {
        ws.close();
      }
      out.textContent = '';
      appendMessage('info', 'Starting...');
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'progress') {
            appendMessage('progress', msg.data);
          } else if (msg.type === 'error') {
            appendMessage('error', msg.data);
          } else {
            appendMessage('info', JSON.stringify(msg));
          }
        } catch {
          appendMessage('info', ev.data);
        }
      };
      ws.onclose = () => appendMessage('info', '\\nClosed.');
      ws.onerror = () => appendMessage('error', 'WebSocket error occurred');
      ws.onopen = () => {
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
    };
  </script>
</body>
</html>
""")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal:
            await websocket.send_json({"type": "error", "data": "No goal provided"})
            await websocket.close()
            return

        def on_progress(message: str) -> None:
            clean = _strip_rich_markup(message)
            asyncio.create_task(
                websocket.send_json({"type": "progress", "data": clean})
            )

        orchestrator = Orchestrator(goal=goal, on_progress=on_progress)
        await orchestrator.run()
        await websocket.send_json({"type": "progress", "data": "--- Done ---"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    configure_logging(settings.log_level)
    uvicorn.run(app, host=host, port=port)
