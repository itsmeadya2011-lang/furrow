from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

active_orchestrators: list[Orchestrator] = []


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "active_cycles": str(len(active_orchestrators))}


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
    orchestrator: Orchestrator | None = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        orchestrator = Orchestrator(goal=goal)
        active_orchestrators.append(orchestrator)
        try:
            await orchestrator.run()
        except Exception as e:
            await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        if orchestrator is not None and orchestrator in active_orchestrators:
            active_orchestrators.remove(orchestrator)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
