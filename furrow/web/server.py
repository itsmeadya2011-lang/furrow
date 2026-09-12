from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from furrow.config import Provider, Settings, TaskStatus, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent

app = FastAPI(title="Furrow")


async def send_event(websocket: WebSocket, event_type: str, message: str = "", data: dict | None = None) -> bool:
    payload = {"type": event_type, "message": message}
    if data:
        payload["data"] = data
    try:
        await websocket.send_json(payload)
        return True
    except Exception:
        return False


# NOTE(on disconnect / cancellation): The stream runs inline in the WebSocket
# handler and relies solely on send_event() returning False to detect that the
# client has gone away (a client drop is only surfaced as WebSocketDisconnect
# on receive, which we never call concurrently here). Because of this bounded-send
# model, long-running awaits (plan/test/worker) cannot be eagerly cancelled on
# disconnect. Eager cancellation would require a concurrent WebSocket receive,
# which is unsafe here, so we leave the graceful bail-on-send-failure behavior
# in place rather than introducing concurrent receives.
async def run_orchestrator_stream(websocket: WebSocket, goal: str, settings: Settings | None) -> None:
    orchestrator = Orchestrator(goal=goal, settings=settings)
    try:
        if not await send_event(websocket, "cycle", f"Starting Furrow: {goal}", {"goal": goal}):
            return

        while not orchestrator._cycle_limit_reached():
            orchestrator.cycles += 1
            if not await send_event(websocket, "cycle", f"Cycle {orchestrator.cycles} started", {"cycle": orchestrator.cycles}):
                return

            planning_goal = orchestrator._get_planning_goal()
            if not await send_event(websocket, "plan", "Planning...", {"goal": planning_goal}):
                return

            try:
                plan = await orchestrator.planner.plan(planning_goal)
            except Exception as exc:
                await send_event(websocket, "error", f"Planning failed: {exc}")
                orchestrator._record_planner_failure(exc)
                await send_event(websocket, "complete", "Furrow halted due to planning failure", {"halted": True})
                return

            orchestrator._merge_plan(plan)
            if not await send_event(
                websocket,
                "plan",
                "Plan updated",
                {"plan": plan.model_dump(), "tasks": [t.model_dump() for t in orchestrator.tasks]},
            ):
                return

            pending_tasks = [task for task in orchestrator.tasks if task.status == TaskStatus.PENDING]
            if pending_tasks:
                if not await send_event(websocket, "task", f"Executing {len(pending_tasks)} tasks in parallel"):
                    return

                semaphore = asyncio.Semaphore(orchestrator.settings.max_parallel_tasks)

                async def run_task_with_events(task: TaskModel) -> None:
                    async with semaphore:
                        task.status = TaskStatus.RUNNING
                        if not await send_event(websocket, "task", f"Task {task.id} started", {"task": task.model_dump()}):
                            return
                        try:
                            result = await WorkerAgent(task=task, client=orchestrator.client).run()
                        except Exception as exc:
                            orchestrator._fail_task(task, exc)
                            if not await send_event(websocket, "task", f"Task {task.id} failed: {exc}", {"task": task.model_dump()}):
                                return
                        else:
                            task.status = TaskStatus.COMPLETED
                            task.result = str(result)
                            if not await send_event(
                                websocket,
                                "task",
                                f"Task {task.id} completed",
                                {"task": task.model_dump(), "result": task.result},
                            ):
                                return

                results = await asyncio.gather(
                    *(run_task_with_events(task) for task in pending_tasks),
                    return_exceptions=True,
                )
                for task, result in zip(pending_tasks, results):
                    if isinstance(result, Exception):
                        orchestrator._fail_task(task, result)

            if not await send_event(websocket, "test", "Running tests..."):
                return
            try:
                test_result = await TesterAgent(client=orchestrator.client).run(
                    orchestrator.original_goal, orchestrator.tasks
                )
            except Exception as exc:
                test_result = TestResult(passed=False, summary=str(exc), failures=[str(exc)])
                orchestrator._halt_requested = True

            orchestrator.latest_test_result = test_result

            if test_result.passed:
                if not await send_event(websocket, "test", f"Tests passed: {test_result.summary}", {"passed": True, "summary": test_result.summary}):
                    return
                if not pending_tasks and not orchestrator._tasks_are_complete():
                    orchestrator._halt_requested = True
            else:
                if not await send_event(websocket, "test", f"Tests failed: {test_result.summary}", {"passed": False, "summary": test_result.summary, "failures": test_result.failures}):
                    return
                if not pending_tasks:
                    orchestrator._halt_requested = True
                else:
                    if not await send_event(websocket, "cycle", "Will attempt fix in next cycle"):
                        return

            if orchestrator._is_done():
                await send_event(websocket, "complete", "Goal complete. Halting.", {"completed": True})
                break

            if orchestrator._halt_requested:
                await send_event(websocket, "complete", "Furrow halted because no safe progress is possible.", {"halted": True})
                break

            if orchestrator._cycle_limit_reached():
                await send_event(websocket, "complete", f"Cycle limit of {orchestrator.settings.max_cycles} reached. Halting.", {"halted": True, "cycle_limit": True})
                break

    except Exception as exc:
        await send_event(websocket, "error", f"Orchestrator error: {exc}")
        return


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        #messages { white-space: pre-wrap; background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; min-height: 400px; }
        .event { margin: 0.5rem 0; padding: 0.5rem; border-radius: 3px; }
        .cycle { border-left: 3px solid #4ec9b0; }
        .plan { border-left: 3px solid #9cdcfe; }
        .task { border-left: 3px solid #ce9178; }
        .test.passed { border-left: 3px solid #4ec9b0; }
        .test.failed { border-left: 3px solid #f44747; }
        .complete { border-left: 3px solid #dcdcaa; }
        .error { border-left: 3px solid #f44747; background: #3a1a1a; }
        form { margin-bottom: 1rem; }
        input[type="text"] { width: 60%; padding: 0.5rem; }
        button { padding: 0.5rem 1rem; }
    </style>
</head>
<body>
    <h1>Furrow Web Interface</h1>
    <form id="form">
        <input id="goal" type="text" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
        <input id="model" type="text" placeholder="Model (optional)" />
        <input id="provider" type="text" placeholder="Provider: anthropic/openai/ollama (optional)" />
        <button type="submit">Start</button>
    </form>
    <div id="messages"></div>
    <script>
        const form = document.getElementById('form');
        const messages = document.getElementById('messages');
        form.onsubmit = async (e) => {
            e.preventDefault();
            messages.textContent = '';
            const goal = document.getElementById('goal').value;
            const model = document.getElementById('model').value;
            const provider = document.getElementById('provider').value;
            const ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                const event = JSON.parse(ev.data);
                const div = document.createElement('div');
                div.className = 'event ' + event.type + (event.data?.passed === false ? ' failed' : event.data?.passed === true ? ' passed' : '');
                const time = new Date().toLocaleTimeString();
                let content = '[' + time + '] [' + event.type.toUpperCase() + '] ' + event.message;
                if (event.data) {
                    content += ' ' + JSON.stringify(event.data, null, 2);
                }
                div.textContent = content;
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            };
            ws.onclose = () => {
                const div = document.createElement('div');
                div.className = 'event complete';
                div.textContent = '[' + new Date().toLocaleTimeString() + '] [COMPLETE] Connection closed';
                messages.appendChild(div);
            };
            ws.onerror = (err) => {
                const div = document.createElement('div');
                div.className = 'event error';
                div.textContent = '[ERROR] WebSocket error';
                messages.appendChild(div);
            };
            ws.onopen = () => {
                ws.send(JSON.stringify({goal, model: model || undefined, provider: provider || undefined}));
            };
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
        if not isinstance(data, dict):
            await send_event(websocket, "invalid-request", "Request body must be a JSON object")
            return

        goal = data.get("goal", "")
        if not isinstance(goal, str) or not goal.strip():
            await send_event(websocket, "error", "Goal must be a non-empty string")
            return
        goal = goal.strip()

        provider_str = data.get("provider")
        if provider_str is not None and not isinstance(provider_str, str):
            await send_event(websocket, "error", f"Invalid provider: {provider_str}. Must be one of: anthropic, openai, ollama")
            return
        if provider_str:
            try:
                Provider(provider_str)
            except ValueError:
                await send_event(websocket, "error", f"Invalid provider: {provider_str}. Must be one of: anthropic, openai, ollama")
                return

        settings_kwargs = {}
        if provider_str:
            settings_kwargs["provider"] = Provider(provider_str)
        for key in ("model", "planner_model", "worker_model", "tester_model"):
            val = data.get(key)
            if val is not None:
                if not isinstance(val, str):
                    await send_event(websocket, "error", f"Invalid {key}: must be a string")
                    return
                if val:
                    settings_kwargs[key] = val
        workspace = data.get("workspace")
        if workspace is not None and workspace:
            if not isinstance(workspace, str):
                await send_event(websocket, "error", "Invalid workspace: must be a string")
                return
            workspace_path = Path(workspace).expanduser().resolve()
            if not workspace_path.is_dir():
                await send_event(websocket, "error", f"Invalid workspace: not an existing directory: {workspace_path}")
                return
            settings_kwargs["workspace"] = workspace_path
        max_cycles = data.get("max_cycles")
        if max_cycles is not None:
            if not isinstance(max_cycles, int) or isinstance(max_cycles, bool):
                await send_event(websocket, "error", "Invalid max_cycles: must be an integer")
                return
            if max_cycles < 0:
                await send_event(websocket, "error", "Invalid max_cycles: must be >= 0")
                return
            settings_kwargs["max_cycles"] = max_cycles
        max_parallel_tasks = data.get("max_parallel_tasks")
        if max_parallel_tasks is not None:
            if not isinstance(max_parallel_tasks, int) or isinstance(max_parallel_tasks, bool):
                await send_event(websocket, "error", "Invalid max_parallel_tasks: must be an integer")
                return
            if max_parallel_tasks <= 0:
                await send_event(websocket, "error", "Invalid max_parallel_tasks: must be > 0")
                return
            settings_kwargs["max_parallel_tasks"] = max_parallel_tasks

        settings = Settings(**settings_kwargs) if settings_kwargs else None

        await run_orchestrator_stream(websocket, goal, settings)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await send_event(websocket, "error", f"Server error: {exc}")
        except Exception:
            pass


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
