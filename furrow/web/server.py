from __future__ import annotations

import asyncio

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
        model = data.get("model")

        if model:
            settings = Settings()
            settings.model = model
            client = LLMClient(settings=settings)
        else:
            client = None

        queue: asyncio.Queue[str] = asyncio.Queue()

        def status_callback(msg: str) -> None:
            queue.put_nowait(msg)

        orchestrator = Orchestrator(
            goal=goal,
            client=client,
            status_callback=status_callback,
        )

        async def forward_status() -> None:
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                await websocket.send_text(msg)

        async def run_orchestrator() -> None:
            try:
                await orchestrator.run()
            finally:
                queue.put_nowait(None)

        await asyncio.gather(
            run_orchestrator(),
            forward_status(),
        )
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        await websocket.send_text(f"Error: {e}")
    finally:
        await websocket.close()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
