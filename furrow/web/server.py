from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

app = FastAPI(title="Furrow")
logger = logging.getLogger(__name__)
console = Console()

_job_goals: dict[str, str] = {}
_job_queues: dict[str, asyncio.Queue] = {}


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class _JobWebSocket:
    def __init__(self, queue: asyncio.Queue) -> None:
        self._queue = queue

    async def send_json(self, data: Any) -> None:
        await self._queue.put(data)


class WSOrchestrator(Orchestrator):
    def __init__(self, goal: str, websocket: Any, client: Any = None) -> None:
        super().__init__(goal, client)
        self._ws = websocket

    async def run(self) -> None:
        settings: Settings = self.client.settings
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                console.print(
                    f"[bold yellow]Reached max_cycles={settings.max_cycles}. Halting.[/bold yellow]"
                )
                await self._ws.send_json({"phase": "done", "reason": "max_cycles", "cycles": self.cycles})
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._ws.send_json({"phase": "done", "cycles": self.cycles})
                break

    async def _cycle(self) -> None:
        settings: Settings = self.client.settings
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._ws.send_json({"phase": "planned", "plan": plan.model_dump()})

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._ws.send_json({"phase": "task-done", "tasks": []})
            return

        self.all_tasks.extend(plan.tasks)

        sem = asyncio.Semaphore(settings.max_parallel_tasks)

        async def _run_one(task: Any) -> Any:
            async with sem:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [_run_one(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        await self._ws.send_json({"phase": "task-done", "tasks": [t.model_dump() for t in plan.tasks]})

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self.goal = self.original_goal
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = (
                f"{self.original_goal}\n\nFix failing tests:\n"
                + "\n".join(test_result.failures)
            )

        await self._ws.send_json({"phase": "tested", "passed": test_result.passed, "summary": test_result.summary})


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
        const msg = JSON.parse(ev.data);
        out.textContent += JSON.stringify(msg, null, 2) + '\\n';
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
        orchestrator = WSOrchestrator(goal=goal, websocket=websocket)
        await orchestrator.run()
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


@app.post("/api/start")
async def api_start(request: StartRequest) -> dict:
    job_id = str(uuid.uuid4())
    _job_goals[job_id] = request.goal
    _job_queues[job_id] = asyncio.Queue()
    asyncio.create_task(_run_job(job_id, request.goal))
    return {"job_id": job_id}


async def _run_job(job_id: str, goal: str) -> None:
    queue = _job_queues[job_id]
    ws = _JobWebSocket(queue)
    try:
        orchestrator = WSOrchestrator(goal=goal, websocket=ws)
        await orchestrator.run()
    except Exception:
        logger.exception("Job %s failed", job_id)
    finally:
        await queue.put(None)


@app.websocket("/ws/{job_id}")
async def websocket_job_endpoint(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    if job_id not in _job_goals:
        await websocket.send_json({"error": "Unknown job_id"})
        await websocket.close()
        return
    queue = _job_queues[job_id]
    try:
        while True:
            msg = await queue.get()
            if msg is None:
                break
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for job %s", job_id)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
