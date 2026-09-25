from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <input id="model" placeholder="Model (optional)" />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const colors = {
      goal: '#00ffff',
      cycle_start: '#1f75fe',
      plan: '#8e4585',
      task_completed: '#32cd32',
      task_failed: '#ff4136',
      test_result: '#ff8200',
      no_tasks: '#ffd700',
      done: '#ccff00',
      max_cycles_reformed: '#ff69b4',
      max_cycles_reached: '#dc143c',
      error: '#ff0000'
    };
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const event = JSON.parse(ev.data);
          const color = colors[event.type] || '#ffffff';
          const line = document.createElement('div');
          line.style.color = color;
          line.textContent = '[' + event.type + '] ' + JSON.stringify(event);
          out.appendChild(line);
        } catch {
          out.textContent += ev.data + '\\n';
        }
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onopen = () => {
        ws.send(JSON.stringify({
          goal: document.getElementById('goal').value,
          model: document.getElementById('model').value || undefined
        }));
      };
    };
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        settings = Settings()
        if model:
            settings.model = model
        client = LLMClient(settings=settings)

        async def emit(event: dict[str, Any]) -> None:
            try:
                await websocket.send_json(event)
            except Exception:  # noqa: BLE001, S110
                pass  # Client disconnected or error — keep running

        orchestrator = Orchestrator(
            goal=goal, client=client, settings=settings, on_event=emit
        )
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        await websocket.send_json({"type": "error", "message": str(e)})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
