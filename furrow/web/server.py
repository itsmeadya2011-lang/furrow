from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class WebSocketOrchestrator(Orchestrator):
    """Orchestrator that streams output to a WebSocket."""
    
    def __init__(self, goal: str, websocket: WebSocket, client: LLMClient | None = None) -> None:
        super().__init__(goal=goal, client=client)
        self.websocket = websocket

    async def _send(self, message: str) -> None:
        """Send a message to the WebSocket."""
        try:
            await self.websocket.send_text(message)
        except RuntimeError:
            pass  # Connection closed

    async def run(self) -> None:
        await self._send(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}")
        while True:
            self.cycles += 1
            await self._send(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._send("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        await self._send("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal)
        await self._send(f"Plan: {plan.model_dump()}")

        if not plan.tasks:
            await self._send("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our persistent task list
        for task in plan.tasks:
            existing = next((t for t in self.all_tasks if t.id == task.id), None)
            if existing is None:
                self.all_tasks.append(task)

        await self._send("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        tasks = [
            WorkerAgent(task=task, client=self.client).run()
            for task in plan.tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._send(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._send(f"[green]Task {task.id} completed[/green]")

        await self._send("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._send(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._send(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._send(f"  • {failure}")
            await self._send("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)


# Need to import WorkerAgent and TesterAgent for the WebSocket orchestrator
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        client = LLMClient()
        orchestrator = WebSocketOrchestrator(goal=goal, websocket=websocket, client=client)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
