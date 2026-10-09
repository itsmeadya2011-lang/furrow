from __future__ import annotations

import asyncio

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

logger = structlog.get_logger()


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("websocket_connected", count=len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)
        logger.info("websocket_disconnected", count=len(self.active_connections))

    async def broadcast(self, message: dict[str, object]) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:  # noqa: BLE001, S110
                pass


manager = ConnectionManager()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; }
  #log { background: #1e1e1e; color: #d4d4d4; padding: 15px; height: 400px; overflow-y: auto; border-radius: 4px; }
  .cycle { color: #569cd6; }
  .task-completed { color: #4ec9b0; }
  .task-failed { color: #f44747; }
  .planning { color: #dcdcaa; }
  .testing { color: #ce9178; }
  form { margin-bottom: 20px; }
  input { padding: 8px; width: 70%; }
  button { padding: 8px 16px; background: #0e639c; color: white; border: none; border-radius: 4px; cursor: pointer; }
  button:hover { background: #1177bb; }
  table { width: 100%; border-collapse: collapse; margin-top: 10px; }
  th, td { padding: 8px; text-align: left; border-bottom: 1px solid #333; }
  th { color: #9cdcfe; }
</style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit">Start</button>
  </form>
  <div id="status"></div>
  <table id="tasks">
    <thead>
      <tr><th>ID</th><th>Description</th><th>Status</th><th>Retries</th></tr>
    </thead>
    <tbody></tbody>
  </table>
  <pre id="log"></pre>
  <script>
    const form = document.getElementById('form');
    const log = document.getElementById('log');
    const status = document.getElementById('status');
    const tbody = document.querySelector('#tasks tbody');

    form.onsubmit = async (e) => {
      e.preventDefault();
      log.textContent = '';
      tbody.innerHTML = '';
      status.textContent = 'Connecting...';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        handleMessage(msg);
      };
      ws.onclose = () => status.textContent = 'Disconnected';
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };

    function handleMessage(msg) {
      if (msg.type === 'cycle_start') {
        log.textContent += '\\n' + '='.repeat(50) + '\\n';
        log.textContent += `Cycle ${msg.cycle}\\n`;
        log.textContent += '='.repeat(50) + '\\n';
        status.textContent = `Cycle ${msg.cycle} - Planning...`;
      } else if (msg.type === 'plan') {
        log.textContent += `Plan: ${msg.rationale}\\n`;
        msg.tasks.forEach(t => {
          const row = tbody.insertRow();
          row.innerHTML = `<td>${t.id}</td><td>${t.description}</td><td class="pending">pending</td><td>0</td>`;
          row.dataset.taskId = t.id;
        });
      } else if (msg.type === 'task_start') {
        status.textContent = `Cycle ${msg.cycle} - Executing ${msg.task_id}...`;
      } else if (msg.type === 'task_complete') {
        const row = tbody.querySelector(`[data-task-id="${msg.task_id}"]`);
        if (row) {
          row.cells[2].textContent = 'completed';
          row.cells[2].className = 'task-completed';
          row.cells[3].textContent = msg.retries;
        }
        log.textContent += `✓ Task ${msg.task_id} completed\\n`;
      } else if (msg.type === 'task_failed') {
        const row = tbody.querySelector(`[data-task-id="${msg.task_id}"]`);
        if (row) {
          row.cells[2].textContent = 'failed';
          row.cells[2].className = 'task-failed';
          row.cells[3].textContent = msg.retries;
        }
        log.textContent += `✗ Task ${msg.task_id} failed: ${msg.error}\\n`;
      } else if (msg.type === 'test_result') {
        status.textContent = `Cycle ${msg.cycle} - Testing...`;
        if (msg.passed) {
          log.textContent += `✓ Tests passed: ${msg.summary}\\n`;
        } else {
          log.textContent += `✗ Tests failed: ${msg.summary}\\n`;
          msg.failures.forEach(f => log.textContent += `  - ${f}\\n`);
        }
      } else if (msg.type === 'status') {
        log.textContent += msg.message + '\\n';
      } else if (msg.type === 'complete') {
        status.textContent = 'Goal complete!';
        log.textContent += '\\n' + '='.repeat(50) + '\\n';
        log.textContent += 'GOAL COMPLETE\\n';
      } else if (msg.type === 'error') {
        log.textContent += `ERROR: ${msg.message}\\n`;
      }
      log.scrollTop = log.scrollHeight;
    }
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
        orchestrator = Orchestrator(goal=goal)

        await manager.broadcast({"type": "cycle_start", "cycle": 1})

        while True:
            orchestrator.cycles += 1
            cycle = orchestrator.cycles

            await manager.broadcast({"type": "cycle_start", "cycle": cycle})

            # Plan
            plan = await orchestrator.planner.plan(orchestrator.goal, orchestrator.tasks)
            await manager.broadcast({
                "type": "plan",
                "cycle": cycle,
                "rationale": plan.rationale,
                "tasks": [{"id": t.id, "description": t.description} for t in plan.tasks],
            })

            if not plan.tasks:
                await manager.broadcast({"type": "complete", "message": "No tasks planned. Goal may be complete."})
                break

            # Execute tasks
            sorted_tasks = orchestrator._topological_sort(plan.tasks)
            semaphore = asyncio.Semaphore(orchestrator.config.max_parallel_tasks)

            async def run_task(task: TaskModel, cycle_num: int = cycle, sem: asyncio.Semaphore = semaphore) -> TaskModel:
                async with sem:
                    await manager.broadcast({"type": "task_start", "cycle": cycle_num, "task_id": task.id})
                    try:
                        worker = WorkerAgent(task=task, client=orchestrator.client, settings=orchestrator.config)
                        result = await asyncio.wait_for(worker.run(), timeout=orchestrator.config.task_timeout)
                        task.status = "completed"
                        task.result = result
                        await manager.broadcast({"type": "task_complete", "cycle": cycle_num, "task_id": task.id, "retries": task.retry_count})
                    except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
                        task.status = "failed"
                        task.result = str(e)
                        task.retry_count += 1
                        await manager.broadcast({"type": "task_failed", "cycle": cycle_num, "task_id": task.id, "error": str(e), "retries": task.retry_count})
                    orchestrator.tasks[task.id] = task
                    return task

            await asyncio.gather(*[run_task(task) for task in sorted_tasks])

            # Test
            test_result = await orchestrator.tester.run(orchestrator.goal, plan.tasks)
            await manager.broadcast({
                "type": "test_result",
                "cycle": cycle,
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": test_result.failures,
            })

            if test_result.passed:
                await manager.broadcast({"type": "status", "message": f"Tests passed: {test_result.summary}"})
            else:
                orchestrator.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

            orchestrator._save_state()

            if orchestrator._is_done():
                await manager.broadcast({"type": "complete", "message": "Goal complete!"})
                break

            await manager.broadcast({"type": "cycle_start", "cycle": cycle + 1})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.exception("websocket_error", error=str(e))
        await manager.broadcast({"type": "error", "message": str(e)})
        manager.disconnect(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
