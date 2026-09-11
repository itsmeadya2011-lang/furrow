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
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        h1 { color: #4CAF50; }
        #form { margin-bottom: 1rem; }
        #goal { width: 70%; padding: 0.5rem; font-size: 1rem; }
        button { padding: 0.5rem 1rem; font-size: 1rem; background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; max-height: 60vh; overflow: auto; white-space: pre-wrap; }
        .cycle { color: #4ec9b0; font-weight: bold; }
        .plan { color: #9cdcfe; }
        .task { color: #dcdcaa; }
        .success { color: #4ec9b0; }
        .error { color: #f44747; }
        .warning { color: #ffcc00; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <form id="form">
        <input id="goal" placeholder="Enter goal (e.g., Add JWT auth to the API)" required />
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
                line.className = msg.type || '';
                line.textContent = msg.message;
                out.appendChild(line);
                out.scrollTop = out.scrollHeight;
            };
            ws.onclose = () => {
                const line = document.createElement('div');
                line.className = 'warning';
                line.textContent = '\\nConnection closed.';
                out.appendChild(line);
            };
            ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
        };
    </script>
</body>
</html>
""")


class ProgressCallback:
    """Callback to send progress updates over WebSocket."""
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
    
    async def send(self, msg_type: str, message: str):
        await self.websocket.send_json({"type": msg_type, "message": message})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    callback = ProgressCallback(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        # Custom orchestrator that uses our callback
        client = LLMClient()
        orchestrator = Orchestrator(goal=goal, client=client)
        
        # Override console output to use WebSocket
        original_print = orchestrator._cycle
        
        async def cycle_with_progress():
            await callback.send("cycle", f"\n═══ Cycle {orchestrator.cycles + 1} ═══")
            
            # Planning
            await callback.send("plan", "Planning...")
            plan = await orchestrator.planner.plan(orchestrator.goal, orchestrator.all_tasks)
            await callback.send("plan", f"Plan: {plan.rationale}")
            for task in plan.tasks:
                await callback.send("task", f"  - {task.id}: {task.description}")
            
            if not plan.tasks:
                await callback.send("warning", "No tasks planned. Goal may be complete.")
                return
            
            # Track new tasks
            for task in plan.tasks:
                task.status = "pending"
                orchestrator.all_tasks.append(task)
            
            # Execute tasks
            await callback.send("task", "Executing tasks in parallel...")
            tasks = [
                WorkerAgent(task=task, client=client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for task, result in zip(plan.tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    await callback.send("error", f"Task {task.id} failed: {result}")
                else:
                    task.status = "completed"
                    task.result = result
                    await callback.send("success", f"Task {task.id} completed")
            
            # Testing
            await callback.send("plan", "Testing...")
            from furrow.agents.tester import TesterAgent
            test_result = await TesterAgent(client=client).run(orchestrator.goal, orchestrator.all_tasks)
            
            if test_result.passed:
                await callback.send("success", f"Tests passed: {test_result.summary}")
                if "Fix failing tests" in orchestrator.goal:
                    orchestrator.goal = orchestrator.original_goal
            else:
                await callback.send("error", f"Tests failed: {test_result.summary}")
                for failure in test_result.failures:
                    await callback.send("error", f"  • {failure}")
                await callback.send("warning", "Will attempt fix in next cycle.")
                orchestrator.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
        
        # Import WorkerAgent here to avoid circular import
        from furrow.agents.worker import WorkerAgent
        
        await callback.send("cycle", f"Starting Furrow with goal: {goal}")
        
        while True:
            if orchestrator.cycles >= orchestrator.max_cycles:
                await callback.send("warning", f"Max cycles ({orchestrator.max_cycles}) reached. Halting.")
                break
                
            orchestrator.cycles += 1
            await cycle_with_progress()
            
            if orchestrator._is_done():
                await callback.send("success", "Goal complete. Halting.")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await callback.send("error", f"Error: {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
