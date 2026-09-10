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


class StreamingOrchestrator:
    """Orchestrator that streams output to a WebSocket."""
    
    def __init__(self, goal: str, websocket: WebSocket, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.websocket = websocket
        self.client = client or LLMClient()
        from furrow.agents.planner import PlannerAgent
        from furrow.agents.tester import TesterAgent
        from furrow.agents.worker import WorkerAgent
        from furrow.config import Plan, TestResult, settings
        
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = settings.max_cycles
        self.current_plan: Plan | None = None
        self.all_tasks: list = []

    async def send(self, message: dict) -> None:
        """Send a message to the WebSocket."""
        await self.websocket.send_text(json.dumps(message))

    async def run(self) -> None:
        await self.send({"type": "start", "goal": self.goal})
        while True:
            self.cycles += 1
            await self.send({"type": "cycle_start", "cycle": self.cycles})
            
            if self.max_cycles > 0 and self.cycles > self.max_cycles:
                await self.send({"type": "max_cycles_reached", "max_cycles": self.max_cycles})
                break
            
            await self._cycle()
            if self._is_done():
                await self.send({"type": "complete", "message": "Goal complete"})
                break

    async def _cycle(self) -> None:
        await self.send({"type": "planning"})
        from furrow.config import Plan, TestResult
        
        plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        await self.send({"type": "plan", "plan": plan.model_dump()})

        if not plan.tasks:
            await self.send({"type": "no_tasks", "message": "No tasks planned. Goal may be complete."})
            return

        # Add new tasks to the all_tasks list
        for task in plan.tasks:
            existing = next((t for t in self.all_tasks if t.id == task.id), None)
            if not existing:
                self.all_tasks.append(task)

        await self.send({"type": "executing", "task_count": len(plan.tasks)})
        from furrow.agents.worker import WorkerAgent
        tasks = [
            WorkerAgent(task=task, client=self.client).run()
            for task in plan.tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self.send({"type": "task_failed", "task_id": task.id, "error": str(result)})
            else:
                task.status = "completed"
                task.result = result
                await self.send({"type": "task_completed", "task_id": task.id, "result": result})

        await self.send({"type": "testing"})
        from furrow.agents.tester import TesterAgent
        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self.send({"type": "tests_passed", "summary": test_result.summary})
        else:
            await self.send({"type": "tests_failed", "summary": test_result.summary, "failures": test_result.failures})
            # Append failures to goal for next cycle
            self.goal = f"{self.goal}\n\nPrevious test failures to fix:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self.all_tasks
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eaeaea; }
        h1 { color: #00d9ff; }
        #form { margin-bottom: 20px; }
        #goal { width: 70%; padding: 10px; background: #16213e; border: 1px solid #0f3460; color: #eaeaea; }
        button { padding: 10px 20px; background: #0f3460; color: #00d9ff; border: none; cursor: pointer; }
        button:hover { background: #00d9ff; color: #1a1a2e; }
        #out { background: #16213e; padding: 15px; border-radius: 5px; white-space: pre-wrap; height: 500px; overflow-y: auto; border: 1px solid #0f3460; }
        .cycle { color: #00d9ff; font-weight: bold; margin-top: 10px; }
        .planning { color: #ffd700; }
        .executing { color: #00ff7f; }
        .testing { color: #ff69b4; }
        .task_completed { color: #00ff7f; }
        .task_failed { color: #ff4444; }
        .tests_passed { color: #00ff7f; font-weight: bold; }
        .tests_failed { color: #ff4444; font-weight: bold; }
        .complete { color: #00d9ff; font-weight: bold; font-size: 1.2em; }
        .plan { color: #87ceeb; }
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
            const goal = document.getElementById('goal').value;
            out.textContent = '';
            appendLog('Starting Furrow...', 'cycle');
            const ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                const msg = JSON.parse(ev.data);
                handleMessage(msg);
            };
            ws.onclose = () => appendLog('\\nConnection closed.', 'cycle');
            ws.onerror = (err) => appendLog('\\nWebSocket error: ' + err, 'task_failed');
            ws.send(JSON.stringify({goal}));
        };
        
        function appendLog(text, className = '') {
            const span = document.createElement('span');
            span.className = className;
            span.textContent = text + '\\n';
            out.appendChild(span);
            out.scrollTop = out.scrollHeight;
        }
        
        function handleMessage(msg) {
            switch(msg.type) {
                case 'cycle_start':
                    appendLog(`═══ Cycle ${msg.cycle} ═══`, 'cycle');
                    break;
                case 'planning':
                    appendLog('Planning...', 'planning');
                    break;
                case 'plan':
                    appendLog('Plan: ' + JSON.stringify(msg.plan, null, 2), 'plan');
                    break;
                case 'no_tasks':
                    appendLog(msg.message, 'planning');
                    break;
                case 'executing':
                    appendLog(`Executing ${msg.task_count} tasks in parallel...`, 'executing');
                    break;
                case 'task_completed':
                    appendLog(`  ✓ Task ${msg.task_id} completed: ${msg.result}`, 'task_completed');
                    break;
                case 'task_failed':
                    appendLog(`  ✗ Task ${msg.task_id} failed: ${msg.error}`, 'task_failed');
                    break;
                case 'testing':
                    appendLog('Testing...', 'testing');
                    break;
                case 'tests_passed':
                    appendLog(`Tests passed: ${msg.summary}`, 'tests_passed');
                    break;
                case 'tests_failed':
                    appendLog(`Tests failed: ${msg.summary}`, 'tests_failed');
                    msg.failures.forEach(f => appendLog(`  • ${f}`, 'task_failed'));
                    break;
                case 'max_cycles_reached':
                    appendLog(`Max cycles (${msg.max_cycles}) reached. Halting.`, 'cycle');
                    break;
                case 'complete':
                    appendLog(msg.message, 'complete');
                    break;
            }
        }
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
        orchestrator = StreamingOrchestrator(goal=goal, websocket=websocket)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
