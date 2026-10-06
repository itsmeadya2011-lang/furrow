from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 800px;
      margin: 2rem auto;
      padding: 0 1rem;
      color: #24292e;
    }
    h1 { margin-bottom: 1rem; }
    form {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }
    input[type="text"] {
      flex: 1;
      min-width: 200px;
      padding: 0.5rem;
      font-size: 1rem;
      border: 1px solid #d1d5da;
      border-radius: 6px;
    }
    input#model { flex: 0 0 240px; }
    button {
      padding: 0.5rem 1.25rem;
      font-size: 1rem;
      background: #2da44e;
      color: #fff;
      border: none;
      border-radius: 6px;
      cursor: pointer;
    }
    button:hover { background: #2c974b; }
    pre {
      background: #f6f8fa;
      border: 1px solid #e1e4e8;
      border-radius: 6px;
      padding: 1rem;
      min-height: 200px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-wrap: break-word;
      font-size: 0.875rem;
      line-height: 1.5;
    }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <input id="model" placeholder="Model (optional)" />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\nStarting...\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\n';
      ws.onclose = () => out.textContent += '\nClosed.\n';
      ws.send(JSON.stringify({
        goal: document.getElementById('goal').value,
        model: document.getElementById('model').value || undefined,
      }));
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
        model = data.get("model")

        if model:
            settings.model = model

        async def output_callback(message: str) -> None:
            await websocket.send_text(message)

        orchestrator = Orchestrator(goal=goal, output_callback=output_callback)
        await orchestrator.run()
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        return
    except Exception as exc:
        try:
            await websocket.send_text(f"Error: {exc}")
        except Exception:
            pass
        return


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)