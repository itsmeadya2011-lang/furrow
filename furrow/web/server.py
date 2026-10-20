from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

HTML = """<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <p id="goal"></p>
  <p id="status">idle</p>
  <form id="form">
    <input id="goalInput" placeholder="Enter goal" required />
    <button type="submit" id="btn">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const goalEl = document.getElementById('goal');
    const statusEl = document.getElementById('status');
    const btn = document.getElementById('btn');
    form.onsubmit = async (e) => {
      e.preventDefault();
      const g = document.getElementById('goalInput').value;
      goalEl.textContent = g;
      out.textContent = '';
      statusEl.textContent = 'running';
      btn.disabled = true;
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => { out.textContent += ev.data + '\\n'; };
      ws.onclose = () => { statusEl.textContent = 'finished'; btn.disabled = false; };
      ws.onopen = () => { ws.send(JSON.stringify({goal: g})); };
    };
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_text("connected")
    try:
        data = await websocket.receive_json()
    except Exception:  # noqa: BLE001
        await websocket.send_text("error: invalid payload")
        await websocket.close()
        return
    goal = data.get("goal") if isinstance(data, dict) else None
    if not isinstance(goal, str) or not goal.strip():
        await websocket.send_text("error: missing or invalid goal")
        await websocket.close()
        return
    await websocket.send_text(f"goal: {goal}")

    closed = {"v": False}

    async def on_event(event: str) -> None:
        if closed["v"]:
            return
        try:
            await websocket.send_text(event)
        except Exception:  # noqa: BLE001
            closed["v"] = True

    orchestrator = Orchestrator(goal=goal, on_event=on_event)
    try:
        await orchestrator.run()
    except Exception as exc:  # noqa: BLE001
        try:
            await websocket.send_text(f"error: {exc}")
        except Exception:  # noqa: BLE001, S110
            pass
    else:
        try:
            await websocket.send_text("run finished")
        except Exception:  # noqa: BLE001, S110
            pass
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001, S110
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
