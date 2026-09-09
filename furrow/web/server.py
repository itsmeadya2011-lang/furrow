from __future__ import annotations

import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body {
      font-family: system-ui, -apple-system, sans-serif;
      margin: 2rem;
      background: #f5f5f5;
    }
    h1 {
      color: #333;
    }
    form {
      margin-bottom: 1rem;
    }
    input {
      padding: 0.5rem;
      font-size: 1rem;
      border: 1px solid #ccc;
      border-radius: 4px;
      width: 300px;
    }
    button {
      padding: 0.5rem 1rem;
      font-size: 1rem;
      background: #007bff;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }
    button:hover {
      background: #0056b3;
    }
    #out {
      font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 1rem;
      white-space: pre-wrap;
      word-wrap: break-word;
      min-height: 200px;
      max-height: 600px;
      overflow-y: auto;
    }
  </style>
</head>
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
      ws.onerror = (ev) => out.textContent += '\\nError occurred\\n';
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
        orchestrator = Orchestrator(goal=goal)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("WebSocket error")
        try:
            await websocket.send_text(f"Error: {exc}")
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    logger.info("Starting Furrow server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)
