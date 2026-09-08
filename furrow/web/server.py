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

# Global state for active websocket connections
active_connections: list[WebSocket] = []


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


async def broadcast(message: str) -> None:
    """Broadcast message to all connected WebSocket clients."""
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except Exception:
            pass


class WebSocketOrchestrator(Orchestrator):
    """Orchestrator that broadcasts updates via WebSocket."""

    def __init__(self, goal: str, client: LLMClient | None = None, settings_obj: Settings | None = None) -> None:
        super().__init__(goal=goal, client=client, settings_obj=settings_obj)

    async def run(self) -> None:
        await broadcast(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}")
        while True:
            if self.settings.max_cycles and self.cycles >= self.settings.max_cycles:
                await broadcast(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break
            self.cycles += 1
            await broadcast(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                await broadcast("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        await broadcast("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal)
        await broadcast(f"Plan: {plan.model_dump_json()}")

        if not plan.tasks:
            await broadcast("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        existing_ids = {t.id for t in self.all_tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.all_tasks.append(task)
                existing_ids.add(task.id)

        await broadcast("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        tasks = [
            WorkerAgent(task=task, client=self.client, settings=self.settings).run()
            for task in plan.tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await broadcast(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await broadcast(f"[green]Task {task.id} completed[/green]")

        await broadcast("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self.all_tasks)

        if test_result.passed:
            await broadcast(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await broadcast(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await broadcast(f"  • {failure}")
            await broadcast("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)


# Import here to avoid circular imports
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
        h1 { color: #4ec9b0; }
        #form { margin-bottom: 20px; }
        #goal { width: 70%; padding: 10px; background: #3c3c3c; border: 1px solid #555; color: #d4d4d4; }
        button { padding: 10px 20px; background: #4ec9b0; border: none; color: #1e1e1e; font-weight: bold; cursor: pointer; }
        button:hover { background: #3db8a0; }
        #out { background: #252526; padding: 15px; height: 500px; overflow-y: auto; border: 1px solid #3c3c3c; white-space: pre-wrap; }
        .cycle { color: #9cdcfe; font-weight: bold; }
        .plan { color: #dcdcaa; }
        .task { color: #ce9178; }
        .success { color: #4ec9b0; }
        .error { color: #f44747; }
        .warning { color: #dcdcaa; }
    </style>
</head>
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
        let ws = null;
        
        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value;
            out.textContent += '\\nStarting...\\n';
            
            if (ws) ws.close();
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
            ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
            ws.onopen = () => ws.send(JSON.stringify({goal: goal}));
        };
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    active_connections.append(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        client = LLMClient()
        orchestrator = WebSocketOrchestrator(goal=goal, client=client)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"[red]Error: {e}[/red]")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
