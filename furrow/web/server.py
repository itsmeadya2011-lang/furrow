from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None

    @property
    def sanitized_goal(self) -> str:
        return self.goal.strip()[:4000]


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required maxlength="4000" style="width: 400px;" />
    <input id="model" placeholder="Model override (optional)" style="width: 300px;" />
    <button type="submit">Start</button>
  </form>
  <pre id="out" style="background:#111;color:#eee;padding:12px;border-radius:6px;"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent = 'Starting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => { out.textContent += ev.data + '\\n'; };
      ws.onerror = (ev) => { out.textContent += '\\nWebSocket error\\n'; };
      ws.onclose = () => { out.textContent += '\\nClosed.\\n'; };
      const payload = {
        goal: document.getElementById('goal').value,
        model: document.getElementById('model').value || null,
      };
      ws.send(JSON.stringify(payload));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        raw = await websocket.receive_json()
        try:
            req = StartRequest(**raw)
        except ValidationError as e:
            await websocket.send_text(f"Invalid request: {e}")
            await websocket.close()
            return

        settings = Settings()
        if req.model:
            settings.model = req.model

        orchestrator = Orchestrator(goal=req.sanitized_goal, settings=settings)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
