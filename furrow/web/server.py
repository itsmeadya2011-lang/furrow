from __future__ import annotations

import asyncio
from typing import Optional, Callable, Awaitable

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
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; margin: 20px; background: #1e1e1e; color: #d4d4d4; }
        h1 { color: #4ec9b0; }
        #form { margin-bottom: 20px; }
        input { padding: 10px; width: 60%; background: #3c3c3c; border: 1px solid #555; color: #d4d4d4; }
        button { padding: 10px 20px; background: #0e639c; color: white; border: none; cursor: pointer; }
        button:hover { background: #1177bb; }
        #out { background: #252526; padding: 15px; height: 500px; overflow-y: auto; border: 1px solid #3c3c3c; white-space: pre-wrap; }
        .cycle { color: #4ec9b0; font-weight: bold; }
        .plan { color: #9cdcfe; }
        .task-done { color: #4ec9b0; }
        .task-fail { color: #f44747; }
        .test-pass { color: #4ec9b0; }
        .test-fail { color: #f44747; }
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
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      out.textContent += '\\n> Starting: ' + goal + '\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        let line = '';
        if (msg.type === 'cycle') {
          line = '<span class="cycle">═══ Cycle ' + msg.cycle + ' ═══</span>';
        } else if (msg.type === 'plan') {
          line = '<span class="plan">Plan: ' + msg.rationale + '</span>';
        } else if (msg.type === 'task') {
          const cls = msg.status === 'completed' ? 'task-done' : 'task-fail';
          line = '<span class="' + cls + '">Task ' + msg.id + ': ' + msg.status + '</span>';
          if (msg.result) line += ' - ' + msg.result;
        } else if (msg.type === 'test') {
          const cls = msg.passed ? 'test-pass' : 'test-fail';
          line = '<span class="' + cls + '">Tests: ' + msg.summary + '</span>';
          if (msg.failures && msg.failures.length > 0) {
            line += '\\n' + msg.failures.map(f => '  • ' + f).join('\\n');
          }
        } else if (msg.type === 'log') {
          line = msg.message;
        } else if (msg.type === 'done') {
          line = '<span class="test-pass">Goal complete. Halting.</span>';
        }
        out.innerHTML += line + '\\n';
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.innerHTML += '\\n<span style="color:#888">Connection closed.</span>\\n';
      ws.send(JSON.stringify({goal: goal}));
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
        
        # Create a custom orchestrator that streams to websocket
        async def send_update(msg: dict) -> None:
            await websocket.send_json(msg)
        
        orchestrator = Orchestrator(goal=goal)
        # Monkey-patch console output to send via websocket
        await run_orchestrator_with_streaming(orchestrator, send_update)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "log", "message": f"Error: {e}"})
        except Exception:
            pass


async def run_orchestrator_with_streaming(orchestrator: Orchestrator, send_update: Callable[[dict], Awaitable[None]]) -> None:
    """Run orchestrator with streaming updates to websocket."""
    await send_update({"type": "log", "message": f"Goal: {orchestrator.original_goal}"})
    
    while True:
        orchestrator.cycles += 1
        
        # Check max_cycles limit
        if orchestrator.settings.max_cycles > 0 and orchestrator.cycles > orchestrator.settings.max_cycles:
            await send_update({"type": "log", "message": f"Max cycles ({orchestrator.settings.max_cycles}) reached. Stopping."})
            break
        
        await send_update({"type": "cycle", "cycle": orchestrator.cycles})
        
        # Planning
        await send_update({"type": "log", "message": "Planning..."})
        plan = await orchestrator.planner.plan(orchestrator.current_goal)
        await send_update({"type": "plan", "rationale": plan.rationale, "tasks": [t.model_dump() for t in plan.tasks]})
        
        if not plan.tasks:
            await send_update({"type": "log", "message": "No tasks planned. Goal may be complete."})
            break

        # Respect max_parallel_tasks
        tasks_to_run = plan.tasks[:orchestrator.settings.max_parallel_tasks]
        
        await send_update({"type": "log", "message": f"Executing {len(tasks_to_run)} tasks in parallel..."})
        worker_tasks = [
            WorkerAgent(task=task, client=orchestrator.client, settings=orchestrator.settings).run()
            for task in tasks_to_run
        ]
        results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        cycle_tasks = []
        for task, result in zip(tasks_to_run, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await send_update({"type": "task", "id": task.id, "status": "failed", "result": str(result)})
            else:
                task.status = "completed"
                task.result = result
                await send_update({"type": "task", "id": task.id, "status": "completed", "result": result})
            cycle_tasks.append(task)
            orchestrator.all_tasks.append(task)

        await send_update({"type": "log", "message": "Testing..."})
        test_result = await orchestrator.tester.run(orchestrator.current_goal, orchestrator.all_tasks)
        
        await send_update({
            "type": "test",
            "passed": test_result.passed,
            "summary": test_result.summary,
            "failures": test_result.failures
        })

        if test_result.passed:
            pass  # Already sent via test update
        else:
            orchestrator.current_goal = f"{orchestrator.original_goal}\n\nPrevious failures to fix:\n" + "\n".join(test_result.failures)

        orchestrator.cycle_history.append({
            "cycle": orchestrator.cycles,
            "goal": orchestrator.current_goal,
            "tasks": [t.model_dump() for t in cycle_tasks],
            "test_result": test_result.model_dump(),
        })
        
        if orchestrator._is_done():
            await send_update({"type": "done", "message": "Goal complete. Halting."})
            break


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


# Import WorkerAgent here to avoid circular imports
from furrow.agents.worker import WorkerAgent