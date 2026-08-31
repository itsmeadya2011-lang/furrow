from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

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
    model: Optional[str] = None
    max_cycles: Optional[int] = None
    max_parallel: Optional[int] = None


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
        const data = JSON.parse(ev.data);
        out.textContent += JSON.stringify(data) + '\\n';
        out.scrollTop = out.scrollHeight;
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
    client: LLMClient | None = None
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        max_cycles = data.get("max_cycles")
        max_parallel = data.get("max_parallel")

        async def send_event(event: dict[str, Any]) -> None:
            await websocket.send_text(json.dumps(event))

        client = LLMClient()
        orchestrator = Orchestrator(
            goal=goal,
            client=client,
            max_cycles=max_cycles,
            max_parallel=max_parallel,
            on_event=send_event,
        )
        await orchestrator.run()
        await websocket.send_text(json.dumps({"type": "done"}))
    except WebSocketDisconnect:
        pass
    finally:
        if client is not None:
            await client.aclose()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
