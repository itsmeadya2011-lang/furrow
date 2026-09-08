from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

active_sessions: dict[str, WebSocketProgressHandler] = {}


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class WebSocketProgressHandler:
    def __init__(self, websocket: WebSocket, session_id: str) -> None:
        self._ws = websocket
        self.session_id = session_id

    async def send_json(self, payload: dict[str, Any]) -> None:
        try:
            await self._ws.send_text(json.dumps(payload))
        except Exception:
            pass

    async def send_status(self, message: str) -> None:
        await self.send_json({"type": "status", "message": message})

    async def send_plan(self, data: dict[str, Any]) -> None:
        await self.send_json({"type": "plan", "data": data})

    async def send_task_result(self, task_id: str, status: str, result: str | None = None) -> None:
        payload: dict[str, Any] = {"type": "task_result", "task_id": task_id, "status": status}
        if result is not None:
            payload["result"] = result
        await self.send_json(payload)

    async def send_test_result(self, passed: bool, summary: str, failures: list[str] | None = None) -> None:
        payload: dict[str, Any] = {"type": "test_result", "passed": passed, "summary": summary}
        if failures is not None:
            payload["failures"] = failures
        await self.send_json(payload)

    async def send_cycle(self, cycle: int) -> None:
        await self.send_json({"type": "cycle", "cycle": cycle})

    async def send_complete(self, cycles: int) -> None:
        await self.send_json({"type": "complete", "cycles": cycles})

    async def send_error(self, message: str) -> None:
        await self.send_json({"type": "error", "message": message})


class _WebSocketLogHandler(logging.Handler):
    def __init__(self, handler: WebSocketProgressHandler) -> None:
        super().__init__()
        self._handler = handler

    def emit(self, record: logging.LogRecord) -> None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self._handler.send_status(self.format(record))
                )
        except Exception:
            pass


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: monospace; background: #0f111a; color: #c9d1d9; padding: 2rem; }
    h1 { color: #58a6ff; }
    #form { margin-bottom: 1rem; }
    #goal { padding: 0.5rem; width: 60%; font-family: monospace; background: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; }
    button { padding: 0.5rem 1rem; background: #238636; color: white; border: none; border-radius: 4px; cursor: pointer; }
    button:hover { background: #2ea043; }
    #out { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; white-space: pre-wrap; max-height: 70vh; overflow-y: auto; }
    .status { color: #c9d1d9; }
    .plan { color: #79c0ff; }
    .task_result { color: #a5d6ff; }
    .test_result { color: #ffa657; }
    .cycle { color: #58a6ff; font-weight: bold; }
    .complete { color: #3fb950; font-weight: bold; }
    .error { color: #f85149; }
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
    const typeColors = {
      status: 'status',
      plan: 'plan',
      task_result: 'task_result',
      test_result: 'test_result',
      cycle: 'cycle',
      complete: 'complete',
      error: 'error'
    };
    form.onsubmit = async (e) => {
      e.preventDefault();
      out.textContent += '\\nConnecting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => {
        out.textContent += 'Connected.\\n';
        ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          const cls = typeColors[msg.type] || 'status';
          out.innerHTML += '<span class="' + cls + '">' + JSON.stringify(msg) + '</span>\\n';
        } catch {
          out.textContent += ev.data + '\\n';
        }
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => out.textContent += '\\nClosed.\\n';
      ws.onerror = () => out.textContent += '\\nError.\\n';
    };
  </script>
</body>
</html>""")


@app.get("/sessions")
async def sessions_count() -> dict[str, int]:
    return {"active_sessions": len(active_sessions)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = uuid.uuid4().hex[:8]
    handler = WebSocketProgressHandler(websocket, session_id)
    active_sessions[session_id] = handler
    log_handler = _WebSocketLogHandler(handler)
    log_handler.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(log_handler)

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        await handler.send_status(f"Goal received: {goal}")
        orchestrator = Orchestrator(goal=goal)

        async def patched_run() -> None:
            log = __import__("furrow.core.orchestrator", fromlist=["log"]).log  # type: ignore[attr-defined]

            await handler.send_status("Starting run")
            log.info("ws_run_started", goal=goal, session=session_id)

            while True:
                if orchestrator.settings.max_cycles > 0 and orchestrator.cycles >= orchestrator.settings.max_cycles:
                    await handler.send_status(f"Max cycles ({orchestrator.settings.max_cycles}) reached.")
                    log.info("ws_max_cycles_reached", cycles=orchestrator.cycles, session=session_id)
                    break
                orchestrator.cycles += 1

                await handler.send_cycle(orchestrator.cycles)
                log.info("ws_cycle_started", cycle=orchestrator.cycles, session=session_id)

                plan = await orchestrator.planner.plan(orchestrator.goal)
                orchestrator._current_tasks = plan.tasks
                await handler.send_plan({
                    "tasks": [t.model_dump() for t in plan.tasks],
                    "rationale": plan.rationale,
                })
                log.info("ws_plan_created", cycle=orchestrator.cycles, task_count=len(plan.tasks), session=session_id)

                if not plan.tasks:
                    await handler.send_status("No tasks planned. Goal may be complete.")
                    break

                semaphore = asyncio.Semaphore(orchestrator.settings.max_parallel_tasks)

                async def _run_worker(task: Any) -> str:
                    async with semaphore:
                        return await __import__("furrow.agents.worker", fromlist=["WorkerAgent"]).WorkerAgent(task=task, client=orchestrator.client).run()

                results = await asyncio.gather(*[_run_worker(t) for t in plan.tasks], return_exceptions=True)

                for task, result in zip(plan.tasks, results):
                    if isinstance(result, Exception):
                        task.status = "failed"
                        task.result = str(result)
                        await handler.send_task_result(task.id, "failed", str(result))
                        log.warning("ws_task_failed", cycle=orchestrator.cycles, task_id=task.id, error=str(result), session=session_id)
                    else:
                        task.status = "completed"
                        task.result = result
                        await handler.send_task_result(task.id, "completed")
                        log.info("ws_task_completed", cycle=orchestrator.cycles, task_id=task.id, session=session_id)

                test_result = await __import__("furrow.agents.tester", fromlist=["TesterAgent"]).TesterAgent(client=orchestrator.client).run(orchestrator.goal, plan.tasks)
                await handler.send_test_result(test_result.passed, test_result.summary, test_result.failures)

                if test_result.passed:
                    log.info("ws_tests_passed", cycle=orchestrator.cycles, summary=test_result.summary, session=session_id)
                else:
                    log.warning("ws_tests_failed", cycle=orchestrator.cycles, summary=test_result.summary, failures=test_result.failures, session=session_id)
                    orchestrator.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

                orchestrator._save_state()

                if orchestrator._is_done():
                    await handler.send_complete(orchestrator.cycles)
                    log.info("ws_run_completed", cycles=orchestrator.cycles, session=session_id)
                    break

        await patched_run()
    except WebSocketDisconnect:
        pass
    finally:
        active_sessions.pop(session_id, None)
        logging.getLogger().removeHandler(log_handler)
        try:
            await websocket.close()
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
