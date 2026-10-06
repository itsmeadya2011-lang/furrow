from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.agents.worker import WorkerAgent
from furrow.core.orchestrator import Orchestrator

logger = structlog.get_logger()

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


# HTML template with real-time task dashboard
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
    h1 { color: #00d9ff; }
    .container { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; }
    .panel { background: #16213e; border-radius: 8px; padding: 20px; }
    .task-list { max-height: 600px; overflow-y: auto; }
    .task { padding: 12px; margin: 8px 0; border-radius: 4px; border-left: 4px solid #444; transition: all 0.3s; }
    .task.pending { border-color: #666; background: #1f2a44; }
    .task.running { border-color: #00d9ff; background: #1f3a44; animation: pulse 1.5s infinite; }
    .task.completed { border-color: #00ff88; background: #1f442e; }
    .task.failed { border-color: #ff4444; background: #441f1f; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
    .task-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .task-id { font-weight: bold; color: #00d9ff; }
    .task-status { padding: 2px 8px; border-radius: 12px; font-size: 12px; text-transform: uppercase; }
    .status-pending { background: #666; }
    .status-running { background: #00d9ff; color: #000; }
    .status-completed { background: #00ff88; color: #000; }
    .status-failed { background: #ff4444; }
    .task-desc { color: #ccc; font-size: 14px; }
    .task-files { font-size: 12px; color: #888; margin-top: 4px; }
    .task-result { margin-top: 8px; padding: 8px; background: #0f1a2e; border-radius: 4px; font-size: 12px; color: #aaa; white-space: pre-wrap; max-height: 200px; overflow: auto; }
    .cycle-info { font-size: 18px; font-weight: bold; color: #00d9ff; margin-bottom: 20px; }
    .log { background: #0f1a2e; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 12px; max-height: 400px; overflow-y: auto; color: #888; }
    .log-entry { margin: 4px 0; }
    .log-entry.info { color: #888; }
    .log-entry.success { color: #00ff88; }
    .log-entry.warning { color: #ffaa00; }
    .log-entry.error { color: #ff4444; }
    form { margin-bottom: 20px; }
    input { padding: 10px; width: 70%; background: #1f2a44; border: 1px solid #333; color: #eee; border-radius: 4px; }
    button { padding: 10px 20px; background: #00d9ff; border: none; color: #000; font-weight: bold; border-radius: 4px; cursor: pointer; }
    button:hover { background: #00b8d4; }
    .summary { grid-column: 1 / -1; }
  </style>
</head>
<body>
  <h1>Furrow - Autonomous Coding Agent</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
    <button type="submit">Start</button>
  </form>
  <div class="container">
    <div class="panel">
      <div class="cycle-info" id="cycleInfo">Waiting for goal...</div>
      <div class="task-list" id="taskList"></div>
    </div>
    <div class="panel">
      <div class="summary">
        <h3>Live Log</h3>
        <div class="log" id="log"></div>
      </div>
    </div>
  </div>
  <script>
    const form = document.getElementById('form');
    const taskList = document.getElementById('taskList');
    const cycleInfo = document.getElementById('cycleInfo');
    const log = document.getElementById('log');

    function addLog(message, type = 'info') {
      const entry = document.createElement('div');
      entry.className = 'log-entry ' + type;
      entry.textContent = '[' + new Date().toLocaleTimeString() + '] ' + message;
      log.appendChild(entry);
      log.scrollTop = log.scrollHeight;
    }

    function updateTasks(tasks) {
      taskList.innerHTML = '';
      tasks.forEach(task => {
        const div = document.createElement('div');
        div.className = 'task ' + task.status;
        div.innerHTML = `
          <div class="task-header">
            <span class="task-id">Task ${task.id}</span>
            <span class="task-status status-${task.status}">${task.status}</span>
          </div>
          <div class="task-desc">${task.description}</div>
          <div class="task-files">Files: ${task.files.join(', ') || 'any'}</div>
          ${task.result ? '<div class="task-result">' + task.result + '</div>' : ''}
        `;
        taskList.appendChild(div);
      });
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      if (!goal) return;

      taskList.innerHTML = '';
      log.innerHTML = '';
      addLog('Starting Furrow with goal: ' + goal);
      cycleInfo.textContent = 'Cycle 1 - Planning...';

      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          handleMessage(data);
        } catch {
          addLog(ev.data);
        }
      };
      ws.onclose = () => addLog('Connection closed', 'warning');
      ws.onerror = () => addLog('Connection error', 'error');
      ws.send(JSON.stringify({goal: goal}));
    };

    function handleMessage(data) {
      if (data.type === 'cycle') {
        cycleInfo.textContent = 'Cycle ' + data.cycle + ' - ' + data.phase;
        addLog('Cycle ' + data.cycle + ': ' + data.phase, 'info');
      } else if (data.type === 'plan') {
        addLog('Plan created with ' + data.tasks.length + ' tasks', 'success');
      } else if (data.type === 'tasks_update') {
        updateTasks(data.tasks);
      } else if (data.type === 'task_result') {
        addLog('Task ' + data.task_id + ' ' + data.status, data.status === 'completed' ? 'success' : 'error');
        if (data.result) addLog('Result: ' + data.result.substring(0, 200), 'info');
      } else if (data.type === 'test_result') {
        addLog('Tests: ' + data.summary, data.passed ? 'success' : 'error');
        if (data.failures && data.failures.length) {
          data.failures.forEach(f => addLog('Failure: ' + f, 'error'));
        }
      } else if (data.type === 'done') {
        addLog('Goal complete!', 'success');
        cycleInfo.textContent = 'Complete';
      } else if (data.type === 'log') {
        addLog(data.message, data.level);
      } else {
        addLog(JSON.stringify(data));
      }
    }
  </script>
</body>
</html>
"""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        orchestrator = Orchestrator(goal=goal)
        await _run_with_websocket(orchestrator, websocket)

    except WebSocketDisconnect:
        pass
    except (OSError, RuntimeError) as e:
        await websocket.send_text(json.dumps({"type": "log", "message": f"Error: {e}", "level": "error"}))


async def _run_with_websocket(orchestrator: Orchestrator, websocket: WebSocket) -> None:
    """Run orchestrator with websocket updates."""
    import structlog
    logger = structlog.get_logger()

    # Send initial cycle info
    await websocket.send_text(json.dumps({"type": "cycle", "cycle": 1, "phase": "Planning"}))

    # We need to monkey-patch the orchestrator to send updates
    # For simplicity, we'll run the orchestrator and capture its output
    # In a more complete implementation, we'd use callbacks

    async def send_cycle(cycle: int, phase: str):
        await websocket.send_text(json.dumps({"type": "cycle", "cycle": cycle, "phase": phase}))

    async def send_tasks(tasks):
        task_data = [
            {
                "id": t.id,
                "description": t.description,
                "files": t.files,
                "status": t.status,
                "result": t.result,
            }
            for t in tasks
        ]
        await websocket.send_text(json.dumps({"type": "tasks_update", "tasks": task_data}))

    async def send_task_result(task_id: str, status: str, result: str):
        await websocket.send_text(json.dumps({"type": "task_result", "task_id": task_id, "status": status, "result": result}))

    async def send_test_result(passed: bool, summary: str, failures: list):
        await websocket.send_text(json.dumps({"type": "test_result", "passed": passed, "summary": summary, "failures": failures}))

    async def send_done():
        await websocket.send_text(json.dumps({"type": "done"}))

    async def send_log(message: str, level: str = "info"):
        await websocket.send_text(json.dumps({"type": "log", "message": message, "level": level}))

    # Monkey-patch the orchestrator's methods to send updates
    original_cycle = orchestrator._cycle

    async def patched_cycle():
        orchestrator.cycles += 1
        await send_cycle(orchestrator.cycles, "Planning")

        with status_context("Planning..."):
            plan = await orchestrator.planner.plan(orchestrator.goal)

        await send_log(f"Plan: {plan.rationale}", "success")
        await websocket.send_text(json.dumps({"type": "plan", "tasks": len(plan.tasks)}))

        if not plan.tasks:
            await send_log("No tasks planned. Goal may be complete.", "warning")
            return

        # Add new tasks
        for task in plan.tasks:
            if not any(t.id == task.id for t in orchestrator.all_tasks):
                orchestrator.all_tasks.append(task)

        await send_tasks(orchestrator.all_tasks)

        # Execute runnable tasks
        runnable = orchestrator._get_runnable_tasks(plan.tasks)
        if runnable:
            await send_cycle(orchestrator.cycles, "Executing")
            await _execute_with_updates(orchestrator, runnable, send_task_result, send_tasks)

        # Run tests
        await send_cycle(orchestrator.cycles, "Testing")
        test_result = await orchestrator._run_tests_with_updates(send_test_result)

        if test_result.passed:
            await send_log(f"Tests passed: {test_result.summary}", "success")
        else:
            await send_log(f"Tests failed: {test_result.summary}", "error")
            for failure in test_result.failures:
                await send_log(f"Failure: {failure}", "error")
            orchestrator.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        orchestrator._save_state()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def status_context(msg: str):
        yield

    async def _execute_with_updates(orchestrator, tasks, send_task_result, send_tasks):
        max_parallel = min(orchestrator.config.max_parallel_tasks, len(tasks))
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_with_semaphore(task):
            async with semaphore:
                task.status = "running"
                await send_tasks(orchestrator.all_tasks)
                await send_log(f"Starting task {task.id}: {task.description}")

                worker = WorkerAgent(task=task, client=orchestrator.client)
                try:
                    result = await worker.run()
                    task.status = "completed"
                    task.result = result
                    await send_task_result(task.id, "completed", result)
                    await send_log(f"Task {task.id} completed", "success")
                except Exception as e:
                    task.status = "failed"
                    task.result = str(e)
                    await send_task_result(task.id, "failed", str(e))
                    await send_log(f"Task {task.id} failed: {e}", "error")
                await send_tasks(orchestrator.all_tasks)

        await asyncio.gather(*[run_with_semaphore(task) for task in tasks])

    # Add helper method to orchestrator for testing
    async def run_tests_with_updates(send_test_result):
        from furrow.agents.tester import TesterAgent
        test_result = await TesterAgent(client=orchestrator.client).run(orchestrator.goal, orchestrator.all_tasks)
        await send_test_result(test_result.passed, test_result.summary, test_result.failures)
        return test_result

    orchestrator._run_tests_with_updates = run_tests_with_updates

    # Run cycles
    while True:
        if orchestrator.config.max_cycles > 0 and orchestrator.cycles >= orchestrator.config.max_cycles:
            await send_log(f"Max cycles ({orchestrator.config.max_cycles}) reached", "warning")
            break

        await patched_cycle()

        if orchestrator._is_done():
            await send_done()
            break

    await send_log("Orchestrator finished")


# Need to import WorkerAgent
from furrow.agents.worker import WorkerAgent


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)