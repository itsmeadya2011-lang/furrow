from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator, OutputHandler
from furrow.git import GitManager

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; min-height: 400px; white-space: pre-wrap; }
        input { width: 70%; padding: 0.5rem; font-size: 1rem; }
        button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
        .status { color: #4ec9b0; }
        .error { color: #f44747; }
        .warning { color: #dcdcaa; }
        .success { color: #4ec9b0; }
    </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      out.textContent += '\\nStarting: ' + goal + '\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const msg = ev.data;
        if (msg.startsWith('[red]') || msg.includes('failed')) {
          out.innerHTML += '<span class="error">' + msg.replace(/[\\[\\]]/g, '') + '</span>\\n';
        } else if (msg.startsWith('[green]') || msg.includes('passed') || msg.includes('completed')) {
          out.innerHTML += '<span class="success">' + msg.replace(/[\\[\\]]/g, '') + '</span>\\n';
        } else if (msg.startsWith('[yellow]') || msg.startsWith('[bold yellow]')) {
          out.innerHTML += '<span class="warning">' + msg.replace(/[\\[\\]]/g, '') + '</span>\\n';
        } else if (msg.startsWith('[bold cyan]') || msg.startsWith('[bold green]')) {
          out.innerHTML += '<span class="status">' + msg.replace(/[\\[\\]]/g, '') + '</span>\\n';
        } else {
          out.textContent += msg + '\\n';
        }
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
      ws.onopen = () => ws.send(JSON.stringify({goal: goal}));
    };
  </script>
</body>
</html>
"""
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")

        # Override model if provided
        if model:
            settings.model = model

        # Create output handler that streams to websocket
        output_handler = OutputHandler(websocket=websocket)

        git_manager = GitManager(Path(settings.workspace))
        orchestrator = Orchestrator(
            goal=goal, output_handler=output_handler, git_manager=git_manager
        )
        await orchestrator.run()

        # Send final summary
        summary = orchestrator.get_task_summary()
        await websocket.send_text(f"\n[bold green]Final Summary:[/bold green] {summary}")
    except WebSocketDisconnect:
        pass
    except OSError as e:
        await websocket.send_text(f"[red]Error: {e}[/red]")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
