from __future__ import annotations

import asyncio
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


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
    h1 { color: #4CAF50; }
    form { margin-bottom: 1rem; }
    input { padding: 0.5rem; width: 70%; }
    button { padding: 0.5rem 1rem; background: #4CAF50; color: white; border: none; cursor: pointer; }
    button:disabled { background: #ccc; cursor: not-allowed; }
    pre { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; }
    .cycle { color: #64b5f6; font-weight: bold; }
    .task-complete { color: #81c784; }
    .task-failed { color: #e57373; }
    .test-pass { color: #81c784; }
    .test-fail { color: #e57373; }
    .planning { color: #ffb74d; }
    .info { color: #90caf9; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit" id="btn">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const btn = document.getElementById('btn');
    let ws = null;
    
    function addMessage(msg, className = '') {
      const div = document.createElement('div');
      div.className = className;
      div.textContent = msg;
      out.appendChild(div);
      out.scrollTop = out.scrollHeight;
    }
    
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      if (!goal) return;
      
      btn.disabled = true;
      btn.textContent = 'Running...';
      out.textContent = '';
      
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => addMessage(ev.data);
      ws.onclose = () => {
        addMessage('\n--- Session ended ---', 'info');
        btn.disabled = false;
        btn.textContent = 'Start';
      };
      ws.onopen = () => ws.send(JSON.stringify({goal: goal}));
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
        
        async def send_output(message: str) -> None:
            # Strip rich markup for web display
            clean = message.replace('[bold green]', '').replace('[/bold green]', '') \
                          .replace('[bold cyan]', '').replace('[/bold cyan]', '') \
                          .replace('[bold yellow]', '').replace('[/bold yellow]', '') \
                          .replace('[green]', '').replace('[/green]', '') \
                          .replace('[red]', '').replace('[/red]', '') \
                          .replace('[yellow]', '').replace('[/yellow]', '') \
                          .replace('[blue]', '').replace('[/blue]', '') \
                          .replace('[bold]', '').replace('[/bold]', '') \
                          .replace('═══', '===').replace('Panel.fit', '').replace('Panel(', '').replace('Pretty(', '')
            await websocket.send_text(clean)
        
        orchestrator = Orchestrator(goal=goal, on_output=send_output)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"Error: {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
