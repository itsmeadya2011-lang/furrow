from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    provider: str | None = None
    planner_model: str | None = None
    worker_model: str | None = None
    tester_model: str | None = None
    max_cycles: int | None = None
    max_parallel: int | None = None


def _build_client(data: dict) -> LLMClient | None:
    model = data.get("model")
    provider = data.get("provider")
    planner_model = data.get("planner_model")
    worker_model = data.get("worker_model")
    tester_model = data.get("tester_model")
    max_cycles = data.get("max_cycles")
    max_parallel = data.get("max_parallel")

    overrides: dict = {}
    if model is not None:
        overrides["model"] = model
    if provider is not None:
        overrides["provider"] = Provider(provider)
    if planner_model is not None:
        overrides["planner_model"] = planner_model
    if worker_model is not None:
        overrides["worker_model"] = worker_model
    if tester_model is not None:
        overrides["tester_model"] = tester_model
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    if max_parallel is not None:
        overrides["max_parallel_tasks"] = max_parallel

    if not overrides:
        return None
    return LLMClient(settings=Settings(**overrides))


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
        client = _build_client(data)
        orchestrator = Orchestrator(goal=goal, client=client)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
