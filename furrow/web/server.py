from __future__ import annotations

import asyncio
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

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
    <title>Furrow - Autonomous Coding Agent</title>
    <style>
      body { font-family: monospace; margin: 2em; background: #0d1117; color: #e6edf3; }
      h1 { color: #58a6ff; }
      #form { margin-bottom: 1em; }
      #goal { width: 600px; padding: 8px; background: #161b22; color: #e6edf3; border: 1px solid #30363d; }
      button { padding: 8px 16px; background: #238636; color: white; border: none; cursor: pointer; }
      button:hover { background: #2ea043; }
      #events { white-space: pre-wrap; font-size: 14px; line-height: 1.5; }
      .event { border-bottom: 1px solid #21262d; padding: 4px 0; }
      .cycle { color: #58a6ff; font-weight: bold; }
      .success { color: #388bfd; }
      .error { color: #f85149; }
      .warning { color: #d29922; }
    </style>
  </head>
  <body>
    <h1>Furrow</h1>
    <p>Autonomous coding agent with infinite parallel development loop.</p>
    <form id="form">
      <input id="goal" placeholder="Enter goal" required style="width: 600px;" />
      <button type="submit">Start</button>
    </form>
    <div id="events"></div>
    <script>
      const form = document.getElementById('form');
      const events = document.getElementById('events');
      const addEvent = (type, data) => {
        const div = document.createElement('div');
        div.className = 'event';
        div.innerHTML = '<span class="cycle">[' + type + ']</span> ' + JSON.stringify(data);
        events.appendChild(div);
        events.scrollTop = events.scrollHeight;
      };
      form.onsubmit = async (e) => {
        e.preventDefault();
        events.textContent = '';
        const ws = new WebSocket('ws://' + location.host + '/ws');
        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            addEvent(msg.type, msg.data);
          } catch {
            addEvent('raw', ev.data);
          }
        };
        ws.onclose = () => addEvent('info', 'Connection closed');
        const goal = document.getElementById('goal').value;
        ws.send(JSON.stringify({goal: goal}));
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

        if not goal:
            await websocket.send_json(
                {"type": "error", "data": {"message": "Goal is required"}}
            )
            await websocket.close()
            return

        if model:
            from furrow.config import settings

            settings.model = model

        sentry: dict[str, Any] = {}

        async def on_event(event_type: str, event_data: dict[str, Any]) -> None:
            sentry[event_type] = event_data
            await websocket.send_json(
                {"type": event_type, "data": event_data}
            )

        orchestrator = Orchestrator(goal=goal, on_event=on_event)

        # Run the orchestrator; if the WebSocket disconnects mid-run,
        # cancel the task gracefully.
        task = asyncio.create_task(orchestrator.run())
        try:
            await task
        except asyncio.CancelledError:
            pass
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001 - report any unexpected error over the websocket
        await websocket.send_json(
            {"type": "error", "data": {"message": str(e)}}
        )
    finally:
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
