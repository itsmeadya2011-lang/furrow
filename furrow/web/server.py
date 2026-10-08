from __future__ import annotations

import asyncio
from typing import cast

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    provider: str | None = None
    max_cycles: int | None = None
    max_parallel: int | None = None
    cycle_timeout: int | None = None


class OrchestratorManager:
    def __init__(self) -> None:
        self.orchestrator: Orchestrator | None = None
        self.websocket: WebSocket | None = None

    async def send(self, message: str) -> None:
        if self.websocket:
            try:
                await self.websocket.send_text(message)
            except Exception:  # noqa: BLE001, S110
                pass


manager = OrchestratorManager()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 1rem; }
        #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; white-space: pre-wrap; height: 500px; overflow: auto; }
        form { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
        input { flex: 1; padding: 0.5rem; font-family: monospace; }
        button { padding: 0.5rem 1rem; background: #007acc; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #005a9e; }
        .cycle { border-left: 3px solid #007acc; padding-left: 1rem; margin: 1rem 0; }
        .task { margin-left: 2rem; }
        .completed { color: #4ec9b0; }
        .failed { color: #f44747; }
        .in_progress { color: #dcdcaa; }
    </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <select id="provider">
      <option value="anthropic">Anthropic</option>
      <option value="openai">OpenAI</option>
    </select>
    <input id="max_cycles" type="number" placeholder="Max cycles (0=unlimited)" style="width: 100px;" />
    <button type="submit">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    let ws = null;

    form.onsubmit = async (e) => {
      e.preventDefault();
      if (ws) ws.close();
      out.textContent += '\\n--- Starting new session ---\\n';
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\n--- Session ended ---\\n';
      ws.send(JSON.stringify({
        goal: document.getElementById('goal').value,
        provider: document.getElementById('provider').value,
        max_cycles: parseInt(document.getElementById('max_cycles').value) || 0
      }));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    manager.websocket = websocket
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        provider = data.get("provider")
        max_cycles = data.get("max_cycles")

        config = Settings(
            provider=provider or settings.provider,
            model=data.get("model") or settings.model,
            max_cycles=max_cycles if max_cycles is not None else settings.max_cycles,
            max_parallel_tasks=data.get("max_parallel") or settings.max_parallel_tasks,
            cycle_timeout=data.get("cycle_timeout") or settings.cycle_timeout,
        )

        # Custom console that sends to websocket
        class WSConsole:
            def __init__(self, ws: WebSocket):
                self.ws = ws

            def print(self, message: str) -> None:
                asyncio.create_task(self.ws.send_text(message))

        # Monkey-patch console for this orchestrator
        import furrow.core.orchestrator as orchestrator_module
        original_console = orchestrator_module.console
        orchestrator_module.console = cast(Console, WSConsole(websocket))

        manager.orchestrator = Orchestrator(goal=goal, config=config)
        await manager.orchestrator.run()

        orchestrator_module.console = original_console

    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        await websocket.send_text(f"[red]Error: {e}[/red]")
    finally:
        manager.websocket = None
        manager.orchestrator = None


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)