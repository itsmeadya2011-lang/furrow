from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.websockets import WebSocketState

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
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        async def send_progress(message: str) -> None:
            try:
                if websocket.application_state != WebSocketState.CONNECTED:
                    return
                await websocket.send_text(message)
            except Exception:
                pass

        orchestrator = Orchestrator(goal=goal, on_progress=send_progress)
        await orchestrator.run()
        await websocket.send_text("Furrow finished.")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"[red]Error: {e}[/red]")
        except Exception:
            pass
    finally:
        await websocket.close()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
