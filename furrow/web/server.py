from __future__ import annotations

import asyncio
import logging

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


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
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


class StreamingOrchestrator(Orchestrator):
    """Orchestrator that streams output to a websocket."""
    
    def __init__(self, goal: str, websocket: WebSocket, client: LLMClient | None = None):
        super().__init__(goal, client)
        self.websocket = websocket
    
    async def _send(self, message: str) -> None:
        try:
            await self.websocket.send_text(message)
        except RuntimeError as e:
            logger.debug("websocket_send_failed: error=%s", e)
    
    async def run(self) -> None:
        await self._send(f"🚀 Furrow started\nGoal: {self.goal}")
        while True:
            self.cycles += 1
            await self._send(f"\n{'='*20} Cycle {self.cycles} {'='*20}")
            await self._cycle()
            self._save_state()
            if self._is_done():
                await self._send("✅ Goal complete. Halting.")
                break
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                await self._send(f"⚠️ Max cycles ({settings.max_cycles}) reached. Stopping.")
                break
    
    async def _cycle(self) -> None:
        project_files = self.client.list_files(settings.workspace)
        
        await self._send("📋 Planning...")
        plan = await self.planner.plan(self.goal, project_files, self.all_tasks)
        await self._send(f"Plan: {plan.rationale}\nTasks: {len(plan.tasks)}")
        for task in plan.tasks:
            status = "✓" if task.id in self.completed_task_ids else "⏳"
            await self._send(f"  {status} {task.id}: {task.description} ({', '.join(task.files) or 'any'})")

        if not plan.tasks:
            await self._send("No tasks planned. Goal may be complete.")
            return

        runnable_tasks = self._get_runnable_tasks(plan.tasks)
        if not runnable_tasks:
            await self._send("All tasks completed or blocked on dependencies.")
            return

        await self._send(f"⚡ Executing {len(runnable_tasks)} tasks in parallel...")
        tasks = [
            WorkerAgent(task=task, client=self.client).run()
            for task in runnable_tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(runnable_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._send(f"❌ Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task_result = result if isinstance(result, str) else str(result)
                task.result = task_result
                self.completed_task_ids.add(task.id)
                await self._send(f"✅ Task {task.id} completed: {task_result[:200]}")

        # Update our task list with new tasks from plan
        for new_task in plan.tasks:
            if not any(t.id == new_task.id for t in self.all_tasks):
                self.all_tasks.append(new_task)
            else:
                for i, t in enumerate(self.all_tasks):
                    if t.id == new_task.id:
                        self.all_tasks[i] = new_task
                        break

        await self._send("🧪 Testing...")
        test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)

        if test_result.passed:
            await self._send(f"✅ Tests passed: {test_result.summary}")
        else:
            await self._send(f"❌ Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                await self._send(f"  • {failure}")
            await self._send("Will attempt fix in next cycle.")
            self.goal = f"Original goal: {self.goal}\n\nFix failing tests:\n" + "\n".join(test_result.failures)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal:
            await websocket.send_text("Error: No goal provided")
            return
        
        orchestrator = StreamingOrchestrator(goal=goal, websocket=websocket)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except (RuntimeError, ValueError) as e:
        logger.exception("websocket_error")
        try:
            await websocket.send_text(f"Error: {e}")
        except RuntimeError:
            pass
    except Exception:
        logger.exception("websocket_unexpected_error: error=%s")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
