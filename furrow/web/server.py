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

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


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


class WebSocketOrchestrator:
    """Wrapper around Orchestrator that streams updates via WebSocket."""

    def __init__(self, goal: str, websocket: WebSocket, settings: Settings | None = None):
        self.goal = goal
        self.websocket = websocket
        self.settings = settings or Settings()
        self.client = None
        self.orchestrator = None

    async def run(self) -> None:
        from furrow.llm import LLMClient
        self.client = LLMClient(settings=self.settings)
        self.orchestrator = Orchestrator(goal=self.goal, client=self.client, settings=self.settings)
        # Override the console output to send via WebSocket
        await self._run_with_streaming()

    async def _run_with_streaming(self) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.pretty import Pretty
        from rich.status import Status
        import io

        # Create a custom console that captures output
        output_buffer = io.StringIO()
        web_console = Console(file=output_buffer, force_terminal=True, width=120)

        async def send_output():
            """Send buffered output to WebSocket."""
            content = output_buffer.getvalue()
            if content:
                await self.websocket.send_text(content)
                output_buffer.seek(0)
                output_buffer.truncate(0)

        # Monkey patch the orchestrator's console
        original_console = self.orchestrator.__class__.console
        self.orchestrator.__class__.console = web_console

        try:
            await self._streaming_run(send_output)
        finally:
            # Restore original console
            self.orchestrator.__class__.console = original_console
            await send_output()

    async def _streaming_run(self, send_output) -> None:
        await self.websocket.send_text(f"🚀 Furrow started\nGoal: {self.goal}\n")
        while True:
            if self.settings.max_cycles > 0 and self.orchestrator.cycles >= self.settings.max_cycles:
                await self.websocket.send_text(f"⚠️ Max cycles ({self.settings.max_cycles}) reached. Halting.\n")
                break
            self.orchestrator.cycles += 1
            await self.websocket.send_text(f"\n{'='*20} Cycle {self.orchestrator.cycles} {'='*20}\n")
            await self._streaming_cycle(send_output)
            self.orchestrator._save_state()
            if self.orchestrator._is_done():
                await self.websocket.send_text("\n✅ Goal complete. Halting.\n")
                if self.orchestrator.state_file.exists():
                    self.orchestrator.state_file.unlink()
                break

    async def _streaming_cycle(self, send_output) -> None:
        await self.websocket.send_text("📋 Planning...\n")
        plan = await self.orchestrator.planner.plan(self.orchestrator.goal)
        await self.websocket.send_text(f"Plan: {plan.model_dump()}\n")

        if not plan.tasks:
            await self.websocket.send_text("⚠️ No tasks planned. Goal may be complete.\n")
            return

        for task in plan.tasks:
            self.orchestrator.all_tasks.append(task)

        await self.websocket.send_text(f"⚡ Executing {len(plan.tasks)} task(s) in parallel...\n")
        await self._execute_tasks_with_streaming(plan.tasks, send_output)

        await self.websocket.send_text("🧪 Testing...\n")
        from furrow.agents.tester import TesterAgent
        test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.orchestrator.goal, plan.tasks)

        if test_result.passed:
            await self.websocket.send_text(f"✅ Tests passed: {test_result.summary}\n")
        else:
            await self.websocket.send_text(f"❌ Tests failed: {test_result.summary}\n")
            for failure in test_result.failures:
                await self.websocket.send_text(f"  • {failure}\n")
            await self.websocket.send_text("⚠️ Will attempt fix in next cycle.\n")
            self.orchestrator.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks_with_streaming(self, tasks, send_output) -> None:
        from furrow.agents.worker import WorkerAgent
        pending_tasks = list(tasks)

        for attempt in range(self.orchestrator.max_retries):
            if not pending_tasks:
                break

            if attempt > 0:
                await self.websocket.send_text(f"🔄 Retry attempt {attempt}/{self.orchestrator.max_retries - 1} for {len(pending_tasks)} failed task(s)\n")

            await self.websocket.send_text(f"Running {len(pending_tasks)} task(s)...\n")
            worker_tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in pending_tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            pending_tasks = []
            for task, result in zip(pending_tasks if attempt > 0 else tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    await self.websocket.send_text(f"❌ Task {task.id} failed: {result}\n")
                    if attempt < self.orchestrator.max_retries - 1:
                        pending_tasks.append(task)
                else:
                    task.status = "completed"
                    task.result = result
                    await self.websocket.send_text(f"✅ Task {task.id} completed\n")


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
        orchestrator = WebSocketOrchestrator(goal=goal, websocket=websocket, settings=settings)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"Error: {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
