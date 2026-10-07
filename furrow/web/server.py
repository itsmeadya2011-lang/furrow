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


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass


manager = ConnectionManager()


async def run_orchestrator_with_stream(goal: str, websocket: WebSocket):
    """Run orchestrator and stream output to websocket."""
    client = LLMClient()
    orchestrator = Orchestrator(goal=goal, client=client)
    
    # Override console print to send to websocket
    original_print = orchestrator.run
    
    async def streaming_run():
        from rich.console import Console
        from rich.panel import Panel
        
        # Send initial message
        await manager.send_message(f"🚀 Starting Furrow\nGoal: {goal}\n{'='*50}", websocket)
        
        while True:
            if Settings().max_cycles > 0 and orchestrator.cycles >= Settings().max_cycles:
                await manager.send_message(f"⚠️ Max cycles ({Settings().max_cycles}) reached. Halting.", websocket)
                break
            orchestrator.cycles += 1
            await manager.send_message(f"\n{'='*20} Cycle {orchestrator.cycles} {'='*20}", websocket)
            
            # Planning
            await manager.send_message("📋 Planning...", websocket)
            plan = await orchestrator.planner.plan(orchestrator.goal)
            await manager.send_message(f"Plan: {len(plan.tasks)} tasks\nRationale: {plan.rationale}", websocket)
            
            if not plan.tasks:
                await manager.send_message("No tasks planned. Goal may be complete.", websocket)
                break
            
            orchestrator.all_tasks.extend(plan.tasks)
            
            # Execute tasks
            await manager.send_message(f"⚡ Executing {len(plan.tasks)} tasks in parallel...", websocket)
            semaphore = asyncio.Semaphore(Settings().max_parallel_tasks)
            
            async def run_with_semaphore(task):
                async with semaphore:
                    return await orchestrator.client.complete(
                        f"Task: {task.description}\nFiles: {', '.join(task.files)}",
                        system="You are a coding agent. Implement the task.",
                        model=orchestrator.client.settings.worker_model
                    )
            
            tasks = [run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for task, result in zip(plan.tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    await manager.send_message(f"❌ Task {task.id} failed: {result}", websocket)
                else:
                    task.status = "completed"
                    task.result = result
                    await manager.send_message(f"✅ Task {task.id} completed", websocket)
            
            # Testing
            await manager.send_message("🧪 Running tests...", websocket)
            test_result = await orchestrator.client.complete(
                f"Goal: {orchestrator.goal}\nTest output: (simulated)",
                system="You are a tester. Return JSON with passed, summary, failures.",
                model=orchestrator.client.settings.tester_model
            )
            await manager.send_message(f"Test result: {test_result}", websocket)
            
            if orchestrator._is_done():
                await manager.send_message("🎉 Goal complete!", websocket)
                break
        
        # Print usage
        usage = orchestrator.client.usage
        await manager.send_message(
            f"\n📊 Token Usage:\n"
            f"  Input: {usage.input_tokens:,}\n"
            f"  Output: {usage.output_tokens:,}\n"
            f"  Cost: ${usage.total_cost:.4f}",
            websocket
        )

    try:
        await streaming_run()
    except Exception as e:
        await manager.send_message(f"Error: {e}", websocket)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; background: #1a1a2e; color: #eee; }
        h1 { color: #4CAF50; }
        #form { margin-bottom: 1rem; }
        input { padding: 0.5rem; width: 70%; background: #16213e; border: 1px solid #444; color: #eee; }
        button { padding: 0.5rem 1rem; background: #4CAF50; border: none; color: white; cursor: pointer; }
        button:hover { background: #45a049; }
        #out { background: #0f0f23; padding: 1rem; border-radius: 4px; min-height: 400px; white-space: pre-wrap; overflow-y: auto; }
        .cycle { color: #2196F3; font-weight: bold; }
        .success { color: #4CAF50; }
        .error { color: #f44336; }
        .warning { color: #FF9800; }
    </style>
</head>
<body>
    <h1>Furrow Web UI</h1>
    <form id="form">
        <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
        <button type="submit">Start</button>
    </form>
    <pre id="out"></pre>
    <script>
        const form = document.getElementById('form');
        const out = document.getElementById('out');
        let ws = null;
        
        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value;
            out.textContent = '';
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                const msg = ev.data;
                let cls = '';
                if (msg.includes('✅') || msg.includes('🎉')) cls = 'success';
                else if (msg.includes('❌') || msg.includes('Error')) cls = 'error';
                else if (msg.includes('⚠️')) cls = 'warning';
                else if (msg.includes('Cycle')) cls = 'cycle';
                out.textContent += msg + '\n';
                out.scrollTop = out.scrollHeight;
            };
            ws.onclose = () => out.textContent += '\n--- Connection closed ---\n';
            ws.onopen = () => ws.send(JSON.stringify({goal}));
        };
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        await run_orchestrator_with_stream(goal, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.send_message(f"Error: {e}", websocket)
        manager.disconnect(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
