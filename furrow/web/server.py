from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Furrow</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }
    header {
      background: linear-gradient(135deg, #06b6d4 0%, #2563eb 100%);
      padding: 24px 32px;
      box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
    }
    header h1 {
      margin: 0;
      font-size: 28px;
      font-weight: 700;
      color: #ffffff;
    }
    header p {
      margin: 4px 0 0;
      font-size: 14px;
      opacity: 0.9;
    }
    main {
      flex: 1;
      max-width: 960px;
      margin: 24px auto;
      width: 100%;
      padding: 0 20px;
    }
    form {
      display: flex;
      gap: 12px;
      margin-bottom: 20px;
    }
    #goal {
      flex: 1;
      padding: 12px 16px;
      border: 1px solid #334159;
      border-radius: 8px;
      background: #1e293b;
      color: #e2e8f0;
      font-size: 15px;
      outline: none;
      transition: border-color 0.2s;
    }
    #goal:focus { border-color: #06b6d4; }
    button {
      padding: 12px 24px;
      border: none;
      border-radius: 8px;
      background: #06b6d4;
      color: #0f172a;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s, transform 0.05s;
    }
    button:hover { background: #22d3ee; }
    button:active { transform: scale(0.97); }
    #out {
      background: #0f172a;
      border: 1px solid #334159;
      border-radius: 8px;
      padding: 16px;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 13px;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-y: auto;
      flex: 1;
      height: 500px;
    }
  </style>
</head>
<body>
  <header>
    <h1>Furrow</h1>
    <p>AI-powered autonomous coding agent</p>
  </header>
  <main>
    <form id="form">
      <input id="goal" placeholder="Enter goal" required />
      <button type="submit">Start</button>
    </form>
    <pre id="out"></pre>
  </main>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
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
        queue: asyncio.Queue[str] = asyncio.Queue()
        orchestrator = Orchestrator(goal=goal, message_queue=queue)

        async def sender() -> None:
            try:
                while True:
                    msg = await queue.get()
                    await websocket.send_text(msg)
            except asyncio.CancelledError:
                pass

        sender_task = asyncio.create_task(sender())
        await orchestrator.run()
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
