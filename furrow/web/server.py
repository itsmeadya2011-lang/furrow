from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class WSMessage(BaseModel):
    type: str
    data: Any = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: monospace; margin: 20px; background: #1a1a1a; color: #e0e0e0; }
    #out { white-space: pre-wrap; background: #2a2a2a; padding: 10px; border-radius: 4px; min-height: 300px; max-height: 80vh; overflow-y: auto; }
    input { width: 70%%; padding: 10px; background: #2a2a2a; color: #e0e0e0; border: 1px solid #444; border-radius: 4px; }
    button { padding: 10px 20px; background: #4a9eff; color: white; border: none; border-radius: 4px; cursor: pointer; }
    button:hover { background: #3a8eef; }
    button:disabled { background: #666; cursor: not-allowed; }
    h1 { color: #4ade80; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal (e.g., 'Add user authentication')" required />
    <button type="submit" id="startBtn">Start</button>
  </form>
  <div id="out"></div>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const startBtn = document.getElementById('startBtn');
    let ws = null;
    
    function append(text, color) {
      const span = document.createElement('span');
      span.style.color = color || '#e0e0e0';
      span.textContent = text + '\\n';
      out.appendChild(span);
      out.scrollTop = out.scrollHeight;
    }
    
    form.onsubmit = async (e) => {
      e.preventDefault();
      if (ws) ws.close();
      out.innerHTML = '';
      append('Starting Furrow...', '#4ade80');
      startBtn.disabled = true;
      
      ws = new WebSocket('ws://' + location.host + '/ws');
      
      ws.onopen = () => {
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
      
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'cycle') {
            append(`═══ Cycle ${msg.data} ═══`, '#06b6d4');
          } else if (msg.type === 'status') {
            append(`[${msg.data}]`, '#fbbf24');
          } else if (msg.type === 'task_complete') {
            append(`✓ Task ${msg.data}`, '#4ade80');
          } else if (msg.type === 'task_failed') {
            append(`✗ Task ${msg.data}`, '#ef4444');
          } else if (msg.type === 'test_result') {
            if (msg.data.passed) {
              append(`Tests passed: ${msg.data.summary}`, '#4ade80');
            } else {
              append(`Tests failed: ${msg.data.summary}`, '#ef4444');
              for (const f of msg.data.failures || []) {
                append(`  • ${f}`, '#f87171');
              }
            }
          } else if (msg.type === 'done') {
            append('Goal complete!', '#4ade80');
            startBtn.disabled = false;
          } else if (msg.type === 'error') {
            append(`Error: ${msg.data}`, '#ef4444');
            startBtn.disabled = false;
          } else {
            append(msg.data, '#e0e0e0');
          }
        } catch {
          append(ev.data, '#e0e0e0');
        }
      };
      
      ws.onclose = () => {
        append('\\nConnection closed.', '#888');
        startBtn.disabled = false;
      };
      
      ws.onerror = () => {
        append('WebSocket error', '#ef4444');
        startBtn.disabled = false;
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
        # Wait for goal from client (after connection opens)
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        if not goal:
            await websocket.send_json({"type": "error", "data": "No goal provided"})
            return
        
        # Create orchestrator with message callback
        orchestrator = StreamingOrchestrator(goal=goal, websocket=websocket)
        await orchestrator.run()
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except:
            pass


class StreamingOrchestrator(Orchestrator):
    """Orchestrator that streams progress via WebSocket."""
    
    def __init__(self, goal: str, websocket: WebSocket, **kwargs: Any) -> None:
        super().__init__(goal, **kwargs)
        self.websocket = websocket
        self._sent_cycle = False
    
    async def _send(self, msg_type: str, data: Any = None) -> None:
        try:
            await self.websocket.send_json({"type": msg_type, "data": data})
        except WebSocketDisconnect:
            pass
    
    async def run(self) -> None:
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        await self._send("status", "Starting Furrow")
        
        cycle_count = 0
        while True:
            cycle_count += 1
            self.cycles += 1
            self._sent_cycle = False
            await self._send("cycle", self.cycles)
            
            await self._cycle()
            
            if self._is_done():
                await self._send("done", "Goal complete")
                break
            
            if self.cycles >= self.max_cycles:
                await self._send("status", f"Max cycles ({self.max_cycles}) reached")
                break

    async def _cycle(self) -> None:
        from rich.status import Status
        
        await self._send("status", "Planning...")
        plan = await self.planner.plan(self.goal)

        if not plan.tasks:
            await self._send("status", "No tasks planned")
            return

        # Filter out already-completed tasks
        existing_ids = {t.id for t in self.all_tasks}
        new_tasks = [t for t in plan.tasks if t.id not in existing_ids]
        self.all_tasks.extend(new_tasks)
        
        if not new_tasks:
            await self._send("status", "All tasks completed, running tests")
        else:
            await self._send("status", f"Executing {len(new_tasks)} tasks...")
            results = await self._execute_tasks(new_tasks)

            for task, result in zip(new_tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    await self._send("task_failed", task.id)
                else:
                    task.status = "completed"
                    task.result = result
                    await self._send("task_complete", task.id)

        await self._send("status", "Testing...")
        test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)
        
        await self._send("test_result", {
            "passed": test_result.passed,
            "summary": test_result.summary,
            "failures": test_result.failures,
        })

        if not test_result.passed:
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
