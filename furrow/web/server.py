from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    provider: str | None = None
    max_cycles: int | None = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
    h1 { color: #4CAF50; }
    form { margin-bottom: 20px; }
    input { padding: 10px; width: 70%; font-size: 16px; }
    button { padding: 10px 20px; font-size: 16px; background: #4CAF50; color: white; border: none; cursor: pointer; }
    button:hover { background: #45a049; }
    #out { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; max-height: 600px; overflow: auto; }
    .cycle { border-left: 3px solid #4CAF50; padding-left: 10px; margin: 10px 0; }
    .task { margin-left: 20px; color: #9cdcfe; }
    .success { color: #4ec9b0; }
    .error { color: #f44747; }
    .warning { color: #dcdcaa; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal (e.g., Add JWT auth to the API)" required />
    <select id="provider">
      <option value="anthropic">Anthropic</option>
      <option value="openai">OpenAI</option>
      <option value="ollama">Ollama (local)</option>
    </select>
    <input id="model" placeholder="Model (optional)" style="width: 20%;" />
    <input id="maxCycles" type="number" placeholder="Max cycles (optional)" style="width: 15%;" min="1" />
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
      ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
      ws.send(JSON.stringify({
        goal: document.getElementById('goal').value,
        provider: document.getElementById('provider').value,
        model: document.getElementById('model').value || null,
        maxCycles: parseInt(document.getElementById('maxCycles').value) || null
      }));
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
        provider = data.get("provider")
        max_cycles = data.get("max_cycles")

        # Create custom settings for this session
        from furrow.config import Provider
        session_settings = Settings(
            provider=Provider(provider) if provider else settings.provider,
            model=model or settings.model,
            max_cycles=max_cycles if max_cycles is not None else settings.max_cycles,
            workspace=settings.workspace,
        )

        client = LLMClient(settings=session_settings)
        orchestrator = Orchestrator(
            goal=goal,
            client=client,
            workspace=session_settings.workspace,
            max_cycles=session_settings.max_cycles,
        )

        # Override console output to send via websocket
        from contextlib import redirect_stderr, redirect_stdout

        class WebSocketWriter:
            def __init__(self, ws: WebSocket):
                self.ws = ws
                self.buffer = ""

            def write(self, s: str):
                self.buffer += s
                if "\n" in self.buffer:
                    lines = self.buffer.split("\n")
                    for line in lines[:-1]:
                        if line:
                            asyncio.create_task(self.ws.send_text(line))
                    self.buffer = lines[-1]

            def flush(self):
                if self.buffer:
                    asyncio.create_task(self.ws.send_text(self.buffer))
                    self.buffer = ""

        writer = WebSocketWriter(websocket)
        with redirect_stdout(writer), redirect_stderr(writer):
            await orchestrator.run()

    except WebSocketDisconnect:
        pass
    except OSError as e:
        await websocket.send_text(f"Error: {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
