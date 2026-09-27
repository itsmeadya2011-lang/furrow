from __future__ import annotations

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
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        let msg;
        try { msg = JSON.parse(ev.data); }
        catch (e) { out.textContent += ev.data + '\\n'; return; }
        if (msg.event === 'cycle_complete') {
          out.textContent += 'Cycle ' + msg.cycle + ' complete: ' + (msg.summary || '') + '\\n';
        } else if (msg.event === 'plan') {
          out.textContent += 'Plan: ' + JSON.stringify(msg.plan, null, 2) + '\\n';
        } else if (msg.event === 'error') {
          out.textContent += '[ERROR] ' + (msg.error || 'unknown error') + '\\n';
        } else if (msg.event === 'done') {
          out.textContent += '[DONE] Goal complete.\\n';
        } else {
          out.textContent += JSON.stringify(msg, null, 2) + '\\n';
        }
      };
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

        async def on_cycle(cycle: int, summary: str) -> None:
            await websocket.send_json(
                {"cycle": cycle, "event": "cycle_complete", "summary": summary}
            )

        orchestrator = Orchestrator(goal=goal, on_cycle=on_cycle)
        await orchestrator.run()
        await websocket.send_json({"event": "done"})
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await websocket.send_json({"event": "error", "error": str(e)})
        except Exception:  # noqa: BLE001, S110
            pass
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001, S110
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
