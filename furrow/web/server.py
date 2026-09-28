from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class StartRequest(BaseModel):
    goal: str
    provider: str | None = None
    model: str | None = None
    max_cycles: int | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize shared LLM client connection pool
    yield
    # Cleanup any shared resources on shutdown


app = FastAPI(title="Furrow", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    <select id="provider">
      <option value="">Default</option>
      <option value="anthropic">Anthropic</option>
      <option value="openai">OpenAI</option>
      <option value="ollama">Ollama</option>
    </select>
    <input id="model" placeholder="Model (optional)" />
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
        const data = ev.data;
        if (data.startsWith('{')) {
          out.textContent += '\\n' + data + '\\n';
        } else {
          out.textContent += data + '\\n';
        }
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onopen = () => {
        ws.send(JSON.stringify({
          goal: document.getElementById('goal').value,
          provider: document.getElementById('provider').value,
          model: document.getElementById('model').value
        }));
      };
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def progress_callback(event: str, data: dict[str, Any]) -> None:
        await queue.put(json.dumps({"event": event, **data}))

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        provider = data.get("provider")
        model = data.get("model")
        max_cycles = data.get("max_cycles")

        settings = Settings()
        if provider:
            try:
                settings.provider = Provider(provider)
            except ValueError:
                await websocket.send_text(json.dumps({"event": "error", "message": f"Unknown provider: {provider}"}))
                return
        if model:
            settings.model = model
        if settings.provider == Provider.OLLAMA and not model:
            pass  # use ollama_model default

        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(
            goal=goal,
            client=client,
            max_cycles=max_cycles,
            on_progress=progress_callback,
        )

        # Start the orchestrator in the background, streaming progress
        async def run_orchestrator():
            try:
                await orchestrator.run()
            finally:
                await queue.put(None)  # Sentinel to stop the streaming loop

        runner = asyncio.create_task(run_orchestrator())

        # Stream progress events to the WebSocket client
        while True:
            msg = await queue.get()
            if msg is None:
                break
            await websocket.send_text(msg)

        # Wait for the orchestrator to finish
        await runner
    except WebSocketDisconnect:
        pass
    finally:
        await queue.put(None)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
