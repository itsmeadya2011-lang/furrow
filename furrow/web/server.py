from __future__ import annotations

import asyncio
import json
from typing import Optional

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


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  h1 { color: #2e7d32; }
  input { width: 70%; padding: 0.5rem; font-size: 1rem; }
  button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
  #out { white-space: pre-wrap; background: #f5f5f5; padding: 1rem; border-radius: 4px; margin-top: 1rem; }
  .green { color: #2e7d32; }
  .red { color: #c62828; }
  .yellow { color: #f9a825; }
  .cyan { color: #00838f; }
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
      out.textContent = '';
      const goal = document.getElementById('goal').value;
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        let color = '';
        if (data.type === 'success') color = 'green';
        else if (data.type === 'error') color = 'red';
        else if (data.type === 'warn') color = 'yellow';
        else if (data.type === 'info') color = 'cyan';
        out.textContent += (color ? `<span class="${color}">` : '') + data.message + (color ? '</span>' : '') + '\\n';
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: goal}));
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
        
        settings = Settings()
        if model:
            settings.model = model
        
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal=goal, client=client)
        
        # Monkey-patch console to stream to websocket
        from furrow.core.orchestrator import console as orchestrator_console
        original_print = orchestrator_console.print
        
        def streaming_print(*args, **kwargs):
            original_print(*args, **kwargs)
            for arg in args:
                message = str(arg)
                if message:
                    try:
                        websocket.send_text(json.dumps({"type": "log", "message": message}))
                    except Exception:
                        pass
        
        orchestrator_console.print = streaming_print
        
        await orchestrator.run()
        
        orchestrator_console.print = original_print
        await websocket.send_text(json.dumps({"type": "success", "message": "Done."}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
