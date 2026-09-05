from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class MessageBroadcaster:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    def add(self, websocket: WebSocket) -> None:
        self._connections.append(websocket)

    def remove(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self._connections:
                self._connections.remove(ws)


broadcaster = MessageBroadcaster()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Furrow</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem;
    }
    h1 {
      font-size: 2rem;
      margin-bottom: 1.5rem;
      color: #38bdf8;
    }
    #form {
      display: flex;
      gap: 0.5rem;
      width: 100%;
      max-width: 600px;
      margin-bottom: 1.5rem;
    }
    #goal {
      flex: 1;
      padding: 0.75rem 1rem;
      border: 1px solid #334155;
      border-radius: 0.5rem;
      background: #1e293b;
      color: #e2e8f0;
      font-size: 1rem;
      outline: none;
    }
    #goal:focus { border-color: #38bdf8; }
    button {
      padding: 0.75rem 1.5rem;
      border: none;
      border-radius: 0.5rem;
      background: #38bdf8;
      color: #0f172a;
      font-weight: 600;
      cursor: pointer;
      font-size: 1rem;
    }
    button:hover { background: #7dd3fc; }
    #container {
      width: 100%;
      max-width: 800px;
      height: 60vh;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 0.5rem;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    #header {
      padding: 0.75rem 1rem;
      background: #334155;
      font-weight: 600;
      font-size: 0.875rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    #out {
      flex: 1;
      padding: 1rem;
      overflow-y: auto;
      font-family: "Fira Code", "SF Mono", Menlo, monospace;
      font-size: 0.875rem;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .status { color: #fbbf24; }
    .plan { color: #a78bfa; }
    .error { color: #f87171; }
    .success { color: #4ade80; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required autocomplete="off" />
    <button type="submit">Start</button>
  </form>
  <div id="container">
    <div id="header">Output</div>
    <pre id="out"></pre>
  </div>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');

    function append(data, className) {
      const line = document.createElement('div');
      line.className = className || '';
      line.textContent = data;
      out.appendChild(line);
      out.scrollTop = out.scrollHeight;
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(protocol + '//' + location.host + '/ws');
      ws.onopen = () => {
        append('Connected. Starting...', 'status');
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'status') append(msg.message, 'status');
          else if (msg.type === 'plan') append(JSON.stringify(msg.data, null, 2), 'plan');
          else append(ev.data);
        } catch {
          append(ev.data);
        }
      };
      ws.onerror = () => append('Connection error.', 'error');
      ws.onclose = () => append('Closed.', 'status');
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    broadcaster.add(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        async def status_callback(message: str | dict) -> None:
            if isinstance(message, dict):
                await broadcaster.broadcast(message)
            else:
                await broadcaster.broadcast({"type": "status", "message": message})

        orchestrator = Orchestrator(goal=goal, status_callback=status_callback)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await broadcaster.broadcast({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        broadcaster.remove(websocket)
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
