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


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; padding: 1rem; }
  #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; white-space: pre-wrap; max-height: 70vh; overflow: auto; }
  .event { margin: 0.5rem 0; padding: 0.5rem; border-left: 3px solid #4ec9b0; background: #2d2d2d; }
  .event.cycle { border-color: #569cd6; }
  .event.plan { border-color: #dcdcaa; }
  .event.task { border-color: #4ec9b0; }
  .event.test { border-color: #ce9178; }
  .event.error { border-color: #f44747; background: #3a1a1a; }
  .event.finished { border-color: #6a9955; background: #1a2a1a; }
  .event-type { font-weight: bold; color: #9cdcfe; }
  .event-data { color: #d4d4d4; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required style="width: 60%; padding: 0.5rem;" />
    <button type="submit" style="padding: 0.5rem 1rem;">Start</button>
  </form>
  <div id="out"></div>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value.trim();
      if (!goal) return;
      out.innerHTML += '<div class="event"><span class="event-type">Starting...</span></div>';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          const div = document.createElement('div');
          div.className = 'event ' + (data.type || '');
          if (data.type === 'finished') {
            div.innerHTML = '<span class="event-type">✓ Done</span> (' + data.cycles + ' cycles)';
          } else if (data.type === 'error') {
            div.innerHTML = '<span class="event-type">Error:</span> <span class="event-data">' + data.message + '</span>';
          } else {
            div.innerHTML = '<span class="event-type">' + data.type + ':</span> <span class="event-data">' + JSON.stringify(data, null, 2).replace(/\\n/g, '<br>').replace(/ /g, '&nbsp;') + '</span>';
          }
          out.appendChild(div);
          out.scrollTop = out.scrollHeight;
        } catch {
          out.innerHTML += '<div class="event"><span class="event-data">' + ev.data + '</span></div>';
        }
      };
      ws.onclose = () => {
        const div = document.createElement('div');
        div.className = 'event finished';
        div.textContent = 'Connection closed';
        out.appendChild(div);
      };
      ws.onerror = (err) => {
        const div = document.createElement('div');
        div.className = 'event error';
        div.textContent = 'WebSocket error: ' + err;
        out.appendChild(div);
      };
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
        goal = data.get("goal", "").strip()
        if not goal:
            await websocket.send_json({"type": "error", "message": "Goal cannot be empty"})
            await websocket.close()
            return

        queue: asyncio.Queue[dict] = asyncio.Queue()

        def on_event(event: dict) -> None:
            queue.put_nowait(event)

        orchestrator = Orchestrator(goal=goal, on_event=on_event)
        orchestrator_task = asyncio.create_task(orchestrator.run())

        try:
            while True:
                done, pending = await asyncio.wait(
                    [orchestrator_task, asyncio.create_task(queue.get())],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    if task is orchestrator_task:
                        # Orchestrator finished
                        await websocket.send_json({"type": "finished", "cycles": orchestrator.cycles})
                        return
                    else:
                        # Event received
                        event = task.result()
                        await websocket.send_json(event)
                for task in pending:
                    task.cancel()
        except WebSocketDisconnect:
            orchestrator_task.cancel()
            try:
                await orchestrator_task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            orchestrator_task.cancel()
            try:
                await orchestrator_task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
