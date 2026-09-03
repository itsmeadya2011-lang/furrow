from __future__ import annotations

import asyncio
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


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
      background-color: #1a1a2e;
      color: #e0e0e0;
      font-family: system-ui, -apple-system, sans-serif;
      margin: 0;
      padding: 2rem;
    }
    h1 {
      color: #c9d1d9;
      margin-top: 0;
    }
    form {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1.5rem;
    }
    input[type="text"] {
      flex: 1;
      padding: 0.5rem 0.75rem;
      border: 1px solid #333;
      border-radius: 6px;
      background-color: #0d1117;
      color: #c9d1d9;
      font-size: 1rem;
    }
    button {
      padding: 0.5rem 1.25rem;
      border: none;
      border-radius: 6px;
      background-color: #238636;
      color: #fff;
      font-size: 1rem;
      cursor: pointer;
    }
    button:hover {
      background-color: #2ea043;
    }
    #out {
      background-color: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 1rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.875rem;
      line-height: 1.6;
      white-space: pre-wrap;
      word-wrap: break-word;
      max-height: 70vh;
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
      out.textContent = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        out.textContent += ev.data + '\\n';
      };
      ws.onclose = () => {
        out.textContent += '\\n[connection closed]\\n';
      };
      ws.onerror = (ev) => {
        out.textContent += '\\n[connection error]\\n';
      };
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
        await websocket.send_text(json.dumps({"type": "complete", "message": "Orchestrator run completed."}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
