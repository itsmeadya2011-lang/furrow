from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

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
    <button type="button" id="clear">Clear</button>
  </form>
  <pre id="out" style="max-height: 400px; overflow-y: auto;"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const clearBtn = document.getElementById('clear');
    const scrollToBottom = () => { out.scrollTop = out.scrollHeight; };
    clearBtn.onclick = () => { out.textContent = ''; };
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nStarting...\\n';
      scrollToBottom();
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => { out.textContent += ev.data + '\\n'; scrollToBottom(); };
      ws.onerror = (ev) => { out.textContent += '\\n[error]\\n'; scrollToBottom(); };
      ws.onclose = () => { out.textContent += '\\nClosed.\\n'; scrollToBottom(); };
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
    pending: set[asyncio.Task] = set()

    async def flush_pending() -> None:
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        def on_log(text: str) -> None:
            task = asyncio.create_task(websocket.send_text(text))
            pending.add(task)
            task.add_done_callback(pending.discard)

        orchestrator = Orchestrator(goal=goal, on_log=on_log)
        await orchestrator.run()
        await flush_pending()
        await websocket.send_text("Done")
    except WebSocketDisconnect:
        await flush_pending()
    except Exception as e:
        await flush_pending()
        try:
            await websocket.send_text(f"Error: {e}")
        except WebSocketDisconnect:
            return


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
