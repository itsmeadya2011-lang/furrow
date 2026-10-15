from __future__ import annotations

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


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; height: 60vh; overflow-y: auto; white-space: pre-wrap; }
  .info { color: #9cdcfe; }
  .success { color: #4ec9b0; }
  .warning { color: #dcdcaa; }
  .error { color: #f44747; }
  form { margin: 1rem 0; }
  input { padding: 0.5rem; width: 70%; }
  button { padding: 0.5rem 1rem; }
</style>
</head>
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
        const msg = JSON.parse(ev.data);
        const line = document.createElement('div');
        line.className = msg.level;
        line.textContent = msg.content;
        out.appendChild(line);
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => {
        const line = document.createElement('div');
        line.className = 'warning';
        line.textContent = '\\nClosed.';
        out.appendChild(line);
      };
      ws.onopen = () => {
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
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

        # Create config with optional model override
        config = settings
        if model:
            config = Settings(
                provider=config.provider,
                model=model,
                planner_model=config.planner_model,
                worker_model=config.worker_model,
                tester_model=config.tester_model,
                anthropic_api_key=config.anthropic_api_key,
                openai_api_key=config.openai_api_key,
                ollama_base_url=config.ollama_base_url,
                max_parallel_tasks=config.max_parallel_tasks,
                max_cycles=config.max_cycles,
                workspace=config.workspace,
                log_level=config.log_level,
            )

        client = LLMClient(settings=config)
        orchestrator = Orchestrator(goal=goal, client=client, config=config, websocket=websocket)
        await orchestrator.run()
    except WebSocketDisconnect:
        return
    except BaseException as e:  # noqa: BLE001
        try:
            await websocket.send_json({"level": "error", "content": f"Error: {e}"})
        except OSError:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
