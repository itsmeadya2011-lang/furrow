from __future__ import annotations

import asyncio
import json

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
        content="""<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <input id="max_cycles" type="number" placeholder="Max cycles (0=unlimited)" min="0" />
    <button type="submit">Start</button>
    <button type="button" id="stop" disabled>Stop</button>
  </form>
  <p>Cycle: <span id="cycle">0</span></p>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const cycle = document.getElementById('cycle');
    const stop = document.getElementById('stop');
    let ws = null;
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = '';
      cycle.textContent = '0';
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.event === 'cycle_start') {
          cycle.textContent = msg.data.cycle;
        }
        out.textContent += ev.data + '\\n';
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      stop.disabled = false;
      stop.onclick = () => { if (ws) ws.close(); };
      ws.send(JSON.stringify({
        goal: document.getElementById('goal').value,
        max_cycles: parseInt(document.getElementById('max_cycles').value || '0', 10)
      }));
    };
  </script>
</body>
</html>"""
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()

    def on_event(event_name: str, data: dict) -> None:
        queue.put_nowait((event_name, data))

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        max_cycles = int(data.get("max_cycles", 0) or 0)
        orchestrator = Orchestrator(goal=goal, on_event=on_event)
        run_task = asyncio.create_task(orchestrator.run(max_cycles=max_cycles))
        while not run_task.done():
            send_task = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait(
                [run_task, send_task], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            if run_task in done:
                break
            event_name, event_data = send_task.result()
            await websocket.send_text(json.dumps({"event": event_name, "data": event_data}))
        while not queue.empty():
            event_name, event_data = queue.get_nowait()
            try:
                await websocket.send_text(json.dumps({"event": event_name, "data": event_data}))
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
