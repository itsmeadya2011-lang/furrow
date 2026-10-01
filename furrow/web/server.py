from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

_log = logging.getLogger("furrow.web")

# Track WebSocket tasks to prevent garbage collection
_pending_tasks: set[asyncio.Task] = set()


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body {
      font-family: monospace;
      margin: 2em;
      background: #1a1a1a;
      color: #e0e0e0;
    }
    h1 { color: #4CAF50; }
    input {
      width: 600px;
      padding: 8px;
      background: #2a2a2a;
      color: #e0e0e0;
      border: 1px solid #4CAF50;
    }
    button {
      padding: 8px 16px;
      background: #4CAF50;
      color: #1a1a1a;
      border: none;
      cursor: pointer;
    }
    button:hover { background: #66BB6A; }
    #out {
      white-space: pre-wrap;
      margin-top: 1em;
      background: #0d0d0d;
      padding: 12px;
      border-radius: 4px;
      min-height: 400px;
    }
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
      out.textContent = 'Starting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        out.textContent += ev.data + '\\n';
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onerror = (err) => out.textContent += '\\nError: ' + err + '\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        if not goal:
            await websocket.send_text("Error: No goal provided.")
            return

        settings = Settings()
        if data.get("model"):
            settings.model = data["model"]

        def send_output(text: str) -> None:
            """Forward orchestrator output to the WebSocket client."""
            task = asyncio.create_task(websocket.send_text(text))
            _pending_tasks.add(task)
            task.add_done_callback(_pending_tasks.discard)

        orchestrator = Orchestrator(goal=goal, settings=settings, output_callback=send_output)

        await orchestrator.run()

        await websocket.send_text("\n[Connection closed]")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        with suppress(OSError, RuntimeError):
            await websocket.send_text(f"\nError: {e}")
    finally:
        with suppress(OSError, RuntimeError):
            await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
