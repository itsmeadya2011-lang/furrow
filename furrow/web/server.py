from __future__ import annotations

import asyncio
import json
from typing import Optional

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
    model: Optional[str] = None
    max_cycles: Optional[int] = None


class WebSocketOrchestrator(Orchestrator):
    """Orchestrator that sends updates via WebSocket."""
    
    def __init__(self, goal: str, websocket: WebSocket, client: LLMClient | None = None) -> None:
        super().__init__(goal=goal, client=client)
        self.websocket = websocket
    
    async def _send(self, data: dict) -> None:
        try:
            await self.websocket.send_text(json.dumps(data))
        except Exception:
            pass
    
    async def run(self) -> None:
        await self._send({"type": "start", "goal": self.goal})
        await super().run()
        await self._send({"type": "complete", "summary": self._get_summary()})
    
    async def _cycle(self) -> None:
        await self._send({"type": "cycle_start", "cycle": self.cycles + 1})
        await super()._cycle()
        await self._send({"type": "cycle_end", "cycle": self.cycles})
    
    def _print_summary(self) -> None:
        # Override to send via websocket instead of console
        summary = self._get_summary()
        # Don't print to console in web mode
    
    def _get_summary(self) -> dict:
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        return {
            "cycles": self.cycles,
            "total_tasks": len(self.all_tasks),
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "rationale": self.plan.rationale if self.plan else "",
        }


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
        out.textContent += JSON.stringify(data, null, 2) + '\\n';
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
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        max_cycles = data.get("max_cycles")
        
        if model:
            settings.model = model
        if max_cycles is not None:
            settings.max_cycles = max_cycles
        
        client = LLMClient()
        orchestrator = WebSocketOrchestrator(goal=goal, websocket=websocket, client=client)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
