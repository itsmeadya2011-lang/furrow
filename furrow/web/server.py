from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Furrow</title>
  <style>
    body { font-family: ui-monospace, monospace; background: #1e1e1e; color: #ddd; margin: 0; padding: 1rem; }
    h1 { color: #4CAF50; margin: 0 0 1rem; }
    form { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
    input { flex: 1; padding: 0.5rem; background: #2e2e2e; color: #ddd; border: 1px solid #444; border-radius: 4px; }
    button { padding: 0.5rem 1rem; background: #4CAF50; color: white; border: 0; border-radius: 4px; cursor: pointer; }
    button:disabled { background: #555; cursor: not-allowed; }
    pre { white-space: pre-wrap; word-break: break-word; background: #2e2e2e; padding: 1rem; border-radius: 4px; max-height: 70vh; overflow: auto; }
    .task-completed { color: #8BC34A; }
    .task-failed { color: #f44336; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit" id="btn">Start</button>
  </form>
  <pre id="out">Idle.</pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const btn = document.getElementById('btn');
    const goal = document.getElementById('goal');
    let ws = null;

    form.onsubmit = async (e) => {
      e.preventDefault();
      if (ws) ws.close();
      btn.disabled = true;
      out.textContent = 'Starting...\\n';
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'cycle') {
            out.textContent += `\\n=== Cycle ${msg.cycle} ===\\n`;
          } else if (msg.type === 'plan') {
            out.textContent += `Plan (${msg.num_tasks} tasks): ${msg.rationale}\\n`;
          } else if (msg.type === 'task') {
            const cls = msg.status === 'completed' ? 'task-completed' : 'task-failed';
            out.innerHTML += `<span class="${cls}">Task ${msg.id}: ${msg.status}</span>\\n`;
          } else if (msg.type === 'tests') {
            out.textContent += `Tests ${msg.passed ? 'PASSED' : 'FAILED'}: ${msg.summary}\\n`;
          } else if (msg.type === 'done') {
            out.textContent += `\\nDone. ${msg.reason}\\n`;
          } else if (msg.type === 'error') {
            out.textContent += `ERROR: ${msg.message}\\n`;
          } else {
            out.textContent += ev.data + '\\n';
          }
        } catch {
          out.textContent += ev.data + '\\n';
        }
      };
      ws.onclose = () => {
        out.textContent += '\\nClosed.\\n';
        btn.disabled = false;
      };
      ws.send(JSON.stringify({goal: goal.value}));
    };
  </script>
</body>
</html>
""")


class _OrchestratorEventBus:
    """Forwards Orchestrator lifecycle events to a WebSocket as JSON messages.

    Since the orchestrator runs in the same event loop as the WebSocket
    handler, ``send`` is just an async callable. We schedule sends via
    ``call_later`` so synchronous event handlers can fire-and-forget.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self._send: Callable[[dict[str, Any]], Awaitable[None]] = websocket.send_json

    async def _do_send(self, message: dict) -> None:
        try:
            await self._send(message)
        except Exception:
            # WebSocket likely closed mid-run; swallow so the orchestrator can keep going.
            pass

    def emit_cycle(self, cycle: int) -> None:
        asyncio.ensure_future(self._do_send({"type": "cycle", "cycle": cycle}))

    def emit_plan(self, plan) -> None:
        asyncio.ensure_future(
            self._do_send(
                {
                    "type": "plan",
                    "num_tasks": len(plan.tasks),
                    "rationale": plan.rationale,
                    "tasks": [t.model_dump() for t in plan.tasks],
                }
            )
        )

    def emit_task(self, task_id: str, status: str, result: Optional[str] = None) -> None:
        asyncio.ensure_future(
            self._do_send({"type": "task", "id": task_id, "status": status, "result": result})
        )

    def emit_tests(self, test_result) -> None:
        asyncio.ensure_future(
            self._do_send(
                {
                    "type": "tests",
                    "passed": test_result.passed,
                    "summary": test_result.summary,
                    "failures": test_result.failures,
                }
            )
        )

    def emit_done(self, reason: str) -> None:
        asyncio.ensure_future(self._do_send({"type": "done", "reason": reason}))

    def emit_error(self, message: str) -> None:
        asyncio.ensure_future(self._do_send({"type": "error", "message": message}))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    bus = _OrchestratorEventBus(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal:
            await websocket.send_json({"type": "error", "message": "Empty goal"})
            return

        orchestrator = Orchestrator(goal=goal, event_bus=bus)
        try:
            await orchestrator.run()
        except Exception as e:
            await bus._do_send({"type": "error", "message": str(e)})  # noqa: SLF001
            raise
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)