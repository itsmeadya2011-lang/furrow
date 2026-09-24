from __future__ import annotations

import json

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.agents.worker import WorkerAgent
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    provider: str | None = None
    max_cycles: int | None = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:  # noqa: BLE001
                pass


manager = ConnectionManager()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #2563eb; }
        #form { margin-bottom: 20px; }
        #goal { width: 70%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; background: #2563eb; color: white; border: none; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        #out { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; font-family: monospace; white-space: pre-wrap; max-height: 600px; overflow-y: auto; }
        .cycle { border-left: 3px solid #2563eb; padding-left: 10px; margin: 10px 0; }
        .task { margin-left: 20px; padding: 5px; }
        .task.completed { color: #4ade80; }
        .task.failed { color: #f87171; }
        .task.pending { color: #fbbf24; }
        .test-passed { color: #4ade80; }
        .test-failed { color: #f87171; }
        .summary { margin-top: 20px; padding: 10px; background: #333; border-radius: 8px; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <div id="form">
        <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
        <button type="button" onclick="start()">Start</button>
        <button type="button" onclick="resume()" style="margin-left: 10px;">Resume</button>
    </div>
    <pre id="out"></pre>
    <script>
        const out = document.getElementById('out');
        let ws = null;

        function append(msg) {
            out.textContent += msg + '\\n';
            out.scrollTop = out.scrollHeight;
        }

        function connect(goal, resume) {
            if (ws) ws.close();
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                try {
                    const data = JSON.parse(ev.data);
                    render(data);
                } catch {
                    append(ev.data);
                }
            };
            ws.onclose = () => append('\\n[Connection closed]');
            ws.onopen = () => ws.send(JSON.stringify({goal, resume}));
        }

        function start() {
            const goal = document.getElementById('goal').value;
            if (!goal) return alert('Enter a goal');
            append('\\n[Starting...]\\n');
            connect(goal, false);
        }

        function resume() {
            append('\\n[Resuming...]\\n');
            connect('', true);
        }

        function render(data) {
            if (data.type === 'cycle_start') {
                append(`\\n═══ Cycle ${data.cycle} ═══`);
            } else if (data.type === 'plan') {
                append(`\\n📋 Plan: ${data.rationale}`);
                data.tasks.forEach(t => append(`  • ${t.id}: ${t.description} (${t.files.join(', ') || 'any'})`));
            } else if (data.type === 'task_result') {
                const cls = data.status === 'completed' ? 'completed' : data.status === 'failed' ? 'failed' : 'pending';
                append(`  <span class="task ${cls}">▸ Task ${data.task_id}: ${data.status}</span>`);
            } else if (data.type === 'test_result') {
                const cls = data.passed ? 'test-passed' : 'test-failed';
                append(`\\n  <span class="${cls}">Tests: ${data.passed ? 'PASSED' : 'FAILED'}</span> - ${data.summary}`);
                if (data.failures && data.failures.length) {
                    data.failures.forEach(f => append(`    ✗ ${f}`));
                }
            } else if (data.type === 'done') {
                append(`\\n✅ Goal complete!`);
            } else if (data.type === 'error') {
                append(`\\n❌ Error: ${data.message}`);
            } else if (data.type === 'log') {
                append(data.message);
            }
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
        resume = data.get("resume", False)

        orchestrator = Orchestrator(goal=goal, resume=resume)

        # Override the console output to send via WebSocket
        original_print = orchestrator._cycle

        async def cycle_with_ws():
            await manager.broadcast(json.dumps({"type": "cycle_start", "cycle": orchestrator.cycles + 1}))
            await original_print()

        orchestrator._cycle = cycle_with_ws

        # We need to wrap the run method to capture output
        # For simplicity, run the orchestrator and send updates
        await _run_with_websocket(orchestrator, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        manager.disconnect(websocket)


async def _run_with_websocket(orchestrator: Orchestrator, websocket: WebSocket) -> None:

    # Send initial state
    await websocket.send_text(json.dumps({"type": "log", "message": f"Goal: {orchestrator.goal}"}))

    while True:
        if orchestrator.settings.max_cycles > 0 and orchestrator.cycles >= orchestrator.settings.max_cycles:
            await websocket.send_text(json.dumps({"type": "log", "message": f"Max cycles ({orchestrator.settings.max_cycles}) reached."}))
            break

        orchestrator.cycles += 1
        await websocket.send_text(json.dumps({"type": "cycle_start", "cycle": orchestrator.cycles}))

        # Planning
        plan = await orchestrator.planner.plan(orchestrator.goal, existing_tasks=orchestrator.tasks)

        await websocket.send_text(json.dumps({
            "type": "plan",
            "rationale": plan.rationale,
            "tasks": [{"id": t.id, "description": t.description, "files": t.files} for t in plan.tasks]
        }))

        if not plan.tasks:
            await websocket.send_text(json.dumps({"type": "log", "message": "No tasks planned. Goal may be complete."}))
            break

        orchestrator.tasks = plan.tasks

        # Execute tasks
        import asyncio as asyncio_mod
        semaphore = asyncio_mod.Semaphore(orchestrator.settings.max_parallel_tasks)
        tasks = [
            WorkerAgent(task=task, client=orchestrator.client, settings=orchestrator.settings, semaphore=semaphore).run()
            for task in plan.tasks
        ]
        results = await asyncio_mod.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await websocket.send_text(json.dumps({
                    "type": "task_result",
                    "task_id": task.id,
                    "status": "failed",
                    "message": str(result)
                }))
            else:
                task.status = "completed"
                task.result = result
                await websocket.send_text(json.dumps({
                    "type": "task_result",
                    "task_id": task.id,
                    "status": "completed",
                    "message": result[:200] if len(result) > 200 else result
                }))

        # Testing
        from furrow.agents.tester import TesterAgent
        test_result = await TesterAgent(client=orchestrator.client, settings=orchestrator.settings).run(orchestrator.goal, plan.tasks)

        await websocket.send_text(json.dumps({
            "type": "test_result",
            "passed": test_result.passed,
            "summary": test_result.summary,
            "failures": test_result.failures
        }))

        orchestrator._record_cycle(plan, test_result)
        orchestrator._save_state()

        if test_result.passed:
            await websocket.send_text(json.dumps({"type": "log", "message": f"Tests passed: {test_result.summary}"}))
        else:
            await websocket.send_text(json.dumps({"type": "log", "message": f"Tests failed: {test_result.summary}"}))
            for failure in test_result.failures:
                await websocket.send_text(json.dumps({"type": "log", "message": f"  • {failure}"}))
            orchestrator.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        if orchestrator._is_done():
            await websocket.send_text(json.dumps({"type": "done", "message": "Goal complete!"}))
            orchestrator._save_state(final=True)
            break


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)