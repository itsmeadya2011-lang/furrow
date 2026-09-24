from __future__ import annotations

import asyncio
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent

app = FastAPI(title="Furrow")

# Store active sessions
sessions: dict[str, Orchestrator] = {}


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None
    max_cycles: Optional[int] = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title></head>
<body>
  <h1>Furrow Web Interface</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required style="width: 300px;" />
    <input id="maxCycles" type="number" placeholder="Max cycles (0=unlimited)" value="0" style="width: 200px;" />
    <button type="submit">Start</button>
  </form>
  <hr />
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    let ws = null;
    
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      const maxCycles = parseInt(document.getElementById('maxCycles').value) || 0;
      out.textContent += '\\nStarting...\\n';
      
      if (ws) {
        ws.close();
      }
      
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
      ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
      ws.onopen = () => {
        ws.send(JSON.stringify({goal: goal, max_cycles: maxCycles}));
      };
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        max_cycles = data.get("max_cycles", 0)
        
        # Create a custom settings for this session
        session_settings = Settings()
        if model:
            session_settings.model = model
        if max_cycles:
            session_settings.max_cycles = max_cycles
        
        orchestrator = Orchestrator(goal=goal, settings=session_settings)
        sessions[session_id] = orchestrator
        
        # Run the orchestrator with websocket output
        await _run_with_websocket(orchestrator, websocket)
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"Error: {str(e)}")
    finally:
        sessions.pop(session_id, None)


async def _run_with_websocket(orchestrator: Orchestrator, websocket: WebSocket) -> None:
    """Run orchestrator and send output to websocket."""
    await websocket.send_text(f"[bold green]Furrow[/bold green]\\nGoal: {orchestrator.goal}")
    
    max_cycles = orchestrator.settings.max_cycles
    if max_cycles > 0:
        await websocket.send_text(f"[dim]Max cycles: {max_cycles}[/dim]")

    while True:
        if max_cycles > 0 and orchestrator.cycles >= max_cycles:
            await websocket.send_text(f"[yellow]Reached max cycles ({max_cycles}). Stopping.[/yellow]")
            break

        orchestrator.cycles += 1
        await websocket.send_text(f"\\n[bold cyan]═══ Cycle {orchestrator.cycles} ═══[/bold cyan]")
        
        # Plan
        await websocket.send_text("[bold yellow]Planning...[/bold yellow]")
        plan = await orchestrator.planner.plan(orchestrator.goal)
        await websocket.send_text(f"Plan: {plan.model_dump()}")

        if not plan.tasks:
            await websocket.send_text("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        orchestrator.tasks = plan.tasks
        pending_tasks = [t for t in orchestrator.tasks if t.id not in orchestrator.completed_task_ids and t.id not in orchestrator.failed_task_ids]

        # Execute in batches
        max_parallel = orchestrator.settings.max_parallel_tasks
        for i in range(0, len(pending_tasks), max_parallel):
            batch = pending_tasks[i:i + max_parallel]
            await websocket.send_text(f"[dim]Executing batch {i//max_parallel + 1}: {len(batch)} tasks[/dim]")
            await websocket.send_text("[bold yellow]Executing tasks in parallel...[/bold yellow]")
            
            tasks = [
                WorkerAgent(task=task, client=orchestrator.client, settings=orchestrator.settings).run()
                for task in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    orchestrator.failed_task_ids.add(task.id)
                    await websocket.send_text(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    orchestrator.completed_task_ids.add(task.id)
                    await websocket.send_text(f"[green]Task {task.id} completed[/green]")

        # Test
        await websocket.send_text("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=orchestrator.client, settings=orchestrator.settings).run(orchestrator.goal, orchestrator.tasks)

        if test_result.passed:
            await websocket.send_text(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await websocket.send_text(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await websocket.send_text(f"  • {failure}")
            await websocket.send_text("[yellow]Will attempt fix in next cycle.[/yellow]")
            orchestrator.goal = f"{orchestrator.original_goal}\\n\\nPrevious test failures to fix:\\n" + "\\n".join(test_result.failures)

        orchestrator._save_state()

        if orchestrator._is_done():
            await websocket.send_text("[bold green]Goal complete. Halting.[/bold green]")
            break


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)