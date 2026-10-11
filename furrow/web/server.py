from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.agents.tester import TesterAgent
from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging import setup_logging

# Setup logging
settings = Settings()
setup_logging(settings)

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    resume: bool = False


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
    h1 { color: #4CAF50; }
    #form { margin-bottom: 1rem; }
    #goal { width: 70%; padding: 0.5rem; font-size: 1rem; }
    button { padding: 0.5rem 1rem; font-size: 1rem; background: #4CAF50; color: white; border: none; cursor: pointer; }
    button:hover { background: #45a049; }
    #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; font-family: monospace; white-space: pre-wrap; max-height: 60vh; overflow: auto; }
    .cycle { border-left: 3px solid #2196F3; padding-left: 1rem; margin: 1rem 0; }
    .task { margin-left: 1rem; }
    .task.completed { color: #4CAF50; }
    .task.failed { color: #f44336; }
    .task.pending { color: #ff9800; }
    .log { color: #888; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <label><input type="checkbox" id="resume" /> Resume</label>
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
        try {
          const msg = JSON.parse(ev.data);
          renderMessage(msg);
        } catch {
          out.textContent += ev.data + '\\n';
        }
      };
      ws.onclose = () => out.textContent += '\\n--- Connection closed ---\\n';
      ws.send(JSON.stringify({
        goal: document.getElementById('goal').value,
        resume: document.getElementById('resume').checked
      }));
    };

    function renderMessage(msg) {
      const div = document.createElement('div');
      if (msg.type === 'cycle') {
        div.className = 'cycle';
        div.innerHTML = `<strong>Cycle ${msg.cycle}</strong>: ${msg.goal}`;
      } else if (msg.type === 'plan') {
        div.innerHTML = `<strong>Plan:</strong> ${msg.rationale}<br>` +
          msg.tasks.map(t => `<div class="task ${t.status}">[${t.id}] ${t.description} (${t.status})</div>`).join('');
      } else if (msg.type === 'task') {
        div.className = `task ${msg.status}`;
        div.textContent = `[${msg.task_id}] ${msg.description} - ${msg.status}`;
      } else if (msg.type === 'test') {
        div.style.color = msg.passed ? '#4CAF50' : '#f44336';
        div.textContent = `Tests: ${msg.summary}`;
      } else if (msg.type === 'log') {
        div.className = 'log';
        div.textContent = msg.message;
      } else if (msg.type === 'summary') {
        div.innerHTML = '<strong>Summary:</strong><br>' + Object.entries(msg.data).map(([k,v]) => `${k}: ${v}`).join('<br>');
      } else {
        div.textContent = JSON.stringify(msg);
      }
      out.appendChild(div);
      out.scrollTop = out.scrollHeight;
    }
  </script>
</body>
</html>
"""
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        resume = data.get("resume", False)

        orchestrator = Orchestrator(goal=goal, resume=resume)

        # Monkey-patch to send updates via websocket
        original_cycle = orchestrator._cycle

        async def cycle_with_ws():
            orchestrator.state.cycles += 1
            cycle_num = orchestrator.state.cycles
            await websocket.send_json(
                {"type": "cycle", "cycle": cycle_num, "goal": orchestrator.state.goal}
            )
            await original_cycle()
            # Send summary after cycle
            completed = len(orchestrator.state.completed_tasks)
            failed = len(orchestrator.state.failed_tasks)
            await websocket.send_json(
                {
                    "type": "summary",
                    "data": {
                        "cycle": cycle_num,
                        "completed": completed,
                        "failed": failed,
                    },
                }
            )

        orchestrator._cycle = cycle_with_ws

        # Also patch _execute_batch to send task updates
        original_execute = orchestrator._execute_batch

        async def execute_with_ws(tasks):
            for task in tasks:
                await websocket.send_json(
                    {
                        "type": "task",
                        "task_id": task.id,
                        "description": task.description,
                        "status": "running",
                    }
                )
            completed = await original_execute(tasks)
            # Note: task status updates happen in original_execute
            return completed

        orchestrator._execute_batch = execute_with_ws

        # Patch planner to send plan
        original_plan = orchestrator.planner.plan

        async def plan_with_ws(goal, context):
            plan = await original_plan(goal, context)
            await websocket.send_json(
                {
                    "type": "plan",
                    "rationale": plan.rationale,
                    "tasks": [
                        {"id": t.id, "description": t.description, "status": t.status}
                        for t in plan.tasks
                    ],
                }
            )
            return plan

        orchestrator.planner.plan = plan_with_ws

        # Patch tester to send test results
        original_test = TesterAgent.run

        async def test_with_ws(self, goal, tasks):
            result = await original_test(self, goal, tasks)
            await websocket.send_json(
                {
                    "type": "test",
                    "passed": result.passed,
                    "summary": result.summary,
                    "failures": result.failures,
                }
            )
            return result

        TesterAgent.run = test_with_ws

        await orchestrator.run()

    except WebSocketDisconnect:
        pass
    except (OSError, RuntimeError, ValueError) as e:
        await websocket.send_json({"type": "log", "message": f"Error: {e}"})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
