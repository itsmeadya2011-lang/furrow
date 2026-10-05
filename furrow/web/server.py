from __future__ import annotations

import asyncio

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

log = structlog.get_logger()

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
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
        let line = ev.data;
        try {
          const e = JSON.parse(ev.data);
          line = `[${e.cycle ?? ""}] ${e.type} ${e.task_id ?? e.summary ?? ""}`;
        } catch (_) {}
        out.textContent += line + '\\n';
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
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

        queue: asyncio.Queue[dict] = asyncio.Queue()

        def on_event(event: dict) -> None:
            queue.put_nowait(event)

        async def forward() -> None:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
                if event.get("type") in ("done", "halted"):
                    break

        settings = Settings()
        if model:
            settings.model = model
        orchestrator = Orchestrator(
            goal=goal,
            client=LLMClient(settings=settings),
            on_event=on_event,
        )
        forwarder = asyncio.create_task(forward())
        try:
            await orchestrator.run()
        finally:
            if not forwarder.done():
                queue.put_nowait({"type": "halted", "reason": "error"})
            await forwarder
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001 -- never let a ws error escape the handler
        log.error("websocket_error", error=str(e))


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
