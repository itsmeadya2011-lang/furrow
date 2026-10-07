from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit" id="start">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const start = document.getElementById('start');
    form.onsubmit = async (e) => {
      e.preventDefault();
      start.disabled = true;
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        if (data.type === 'error') {
          out.textContent += '\\n[error] ' + data.message + '\\n';
          out.style.color = 'red';
        } else if (data.type === 'done') {
          out.textContent += '\\n[done] ' + data.reason + '\\n';
        } else {
          out.textContent += '\\n[' + data.type + '] ' + ev.data + '\\n';
        }
      };
      ws.onclose = () => {
        start.disabled = false;
        out.textContent += '\\nClosed.\\n';
      };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


async def _send_events(websocket: WebSocket, queue: asyncio.Queue) -> None:
    while True:
        event = await queue.get()
        if event is None:
            break
        await websocket.send_json(event)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    sender = asyncio.create_task(_send_events(websocket, queue))

    def on_event(event: dict) -> None:
        queue.put_nowait(event)

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal or not goal.strip():
            await websocket.send_json(
                {"type": "error", "message": "goal is required"}
            )
            await websocket.close()
            return
        orchestrator = Orchestrator(goal=goal, on_event=on_event)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await websocket.send_json(
                {"type": "error", "message": str(exc)}
            )
        except Exception:  # noqa: BLE001, S110
            pass
    finally:
        queue.put_nowait(None)
        await sender


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
