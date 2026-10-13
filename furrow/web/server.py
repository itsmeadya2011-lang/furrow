from __future__ import annotations

import asyncio

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
    model: str | None = None


class WebSocketOrchestrator(Orchestrator):
    """Orchestrator that streams output to a WebSocket."""
    
    def __init__(self, goal: str, websocket: WebSocket, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        super().__init__(goal, client, settings)
        self.websocket = websocket
    
    async def _send(self, message: str) -> None:
        try:
            await self.websocket.send_text(message)
        except Exception:
            pass  # Connection closed
    
    async def run(self) -> None:
        await self._send(f"🌱 Furrow started\nGoal: {self.goal}\n")
        while True:
            self.cycles += 1
            
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                await self._send(f"⚠️ Max cycles ({self.settings.max_cycles}) reached. Halting.\n")
                break
                
            await self._send(f"\n{'='*20} Cycle {self.cycles} {'='*20}\n")
            await self._cycle()
            if self._is_done():
                await self._send("\n✅ Goal complete. Halting.\n")
                break

    async def _cycle(self) -> None:
        await self._send("📋 Planning...\n")
        plan = await self.planner.plan(self.goal, self.all_tasks)
        await self._send(f"Plan: {plan.model_dump_json(indent=2)}\n")

        if not plan.tasks:
            await self._send("⚠️ No tasks planned. Goal may be complete.\n")
            return

        self.all_tasks.extend(plan.tasks)

        await self._send(f"⚡ Executing {len(plan.tasks)} tasks in parallel...\n")
        tasks = [
            WorkerAgent(task=task, client=self.client, settings=self.settings).run()
            for task in plan.tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._send(f"❌ Task {task.id} failed: {result}\n")
            else:
                task.status = "completed"
                task.result = result
                await self._send(f"✅ Task {task.id} completed\n")

        await self._send("🧪 Testing...\n")
        test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._send(f"✅ Tests passed: {test_result.summary}\n")
        else:
            await self._send(f"❌ Tests failed: {test_result.summary}\n")
            for failure in test_result.failures:
                await self._send(f"  • {failure}\n")
            await self._send("⚠️ Will attempt fix in next cycle.\n")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)


# Import here to avoid circular imports
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #4CAF50; }
        #form { margin-bottom: 20px; }
        input { width: 70%; padding: 10px; background: #16213e; border: 1px solid #0f3460; color: #eee; }
        button { padding: 10px 20px; background: #4CAF50; border: none; color: white; cursor: pointer; }
        button:hover { background: #45a049; }
        #out { background: #16213e; padding: 15px; border-radius: 5px; min-height: 300px; white-space: pre-wrap; overflow-y: auto; }
        .cycle { color: #2196F3; }
        .success { color: #4CAF50; }
        .error { color: #f44336; }
        .warning { color: #ff9800; }
    </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal (e.g., Add JWT auth to the API)" required />
    <button type="submit">Start</button>
  </form>
  <div id="out"></div>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      out.textContent += '\\n🌱 Starting: ' + goal + '\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const msg = ev.data;
        const div = document.createElement('div');
        if (msg.includes('✅') || msg.includes('✓')) div.className = 'success';
        else if (msg.includes('❌') || msg.includes('✗')) div.className = 'error';
        else if (msg.includes('⚠️') || msg.includes('⚡')) div.className = 'warning';
        else if (msg.includes('Cycle')) div.className = 'cycle';
        div.textContent = msg;
        out.appendChild(div);
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\\n🔌 Connection closed.\\n';
      ws.onerror = (err) => out.textContent += '\\n❌ WebSocket error\\n';
      ws.send(JSON.stringify({goal}));
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
        if not goal:
            await websocket.send_text("Error: No goal provided")
            return
        
        orchestrator = WebSocketOrchestrator(goal=goal, websocket=websocket)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
