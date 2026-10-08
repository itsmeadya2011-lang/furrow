from __future__ import annotations

import asyncio
import sys

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class WebSocketConsole(Console):
    def __init__(self, websocket: WebSocket, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.websocket = websocket

    def print(self, *args, **kwargs):
        super().print(*args, **kwargs)
        try:
            text = self.export_text(styles=False)
            if text.strip():
                asyncio.create_task(self.websocket.send_text(text))
        except Exception:  # noqa: BLE001, S110
            pass


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
        h1 { color: #4ec9b0; }
        form { margin-bottom: 20px; }
        input { padding: 10px; width: 60%; background: #3c3c3c; border: 1px solid #555; color: #d4d4d4; }
        button { padding: 10px 20px; background: #4ec9b0; border: none; color: #1e1e1e; cursor: pointer; }
        button:hover { background: #3db8a0; }
        pre#out { background: #252526; padding: 15px; border-radius: 5px; overflow: auto; max-height: 70vh; white-space: pre-wrap; word-wrap: break-word; }
        .cycle { color: #9cdcfe; font-weight: bold; }
        .plan { color: #ce9178; }
        .task { color: #dcdcaa; }
        .success { color: #4ec9b0; }
        .error { color: #f44747; }
        .warn { color: #dcdcaa; }
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
    let ws = null;
    
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      out.textContent += '\\nStarting...\\n';
      
      if (ws) {
        ws.close();
      }
      
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        out.textContent += ev.data + '\\n';
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
      ws.onerror = (err) => out.textContent += '\\n[Error: ' + err + ']\\n';
      ws.send(JSON.stringify({goal: goal}));
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

        ws_console = WebSocketConsole(websocket, file=sys.stdout, force_terminal=True)

        orchestrator = Orchestrator(goal=goal)
        orchestrator.console = ws_console

        await orchestrator.run()

        await websocket.send_text("\n[Session complete]")
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await websocket.send_text(f"\n[Error: {e}]")
        except Exception:  # noqa: BLE001, S110
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)