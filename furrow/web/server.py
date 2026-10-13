from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

app = FastAPI(title="Furrow")

log = logging.getLogger(__name__)


class StartRequest(BaseModel):
    goal: str
    model: str | None = None
    resume: bool = False


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
            except Exception as e:  # noqa: BLE001
                log.debug("Broadcast failed: %s", e)


manager = ConnectionManager()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        h1 { color: #4CAF50; }
        #form { margin-bottom: 1rem; }
        #goal { width: 70%; padding: 0.5rem; font-size: 1rem; }
        button { padding: 0.5rem 1rem; font-size: 1rem; background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; max-height: 70vh; overflow-y: auto; white-space: pre-wrap; }
        .cycle { color: #569cd6; font-weight: bold; }
        .plan { color: #9cdcfe; }
        .task-done { color: #4ec9b0; }
        .task-fail { color: #f44747; }
        .test-pass { color: #4ec9b0; }
        .test-fail { color: #f44747; }
        .summary { color: #dcdcaa; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <form id="form">
        <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
        <button type="submit">Start</button>
        <label style="margin-left: 1rem;"><input type="checkbox" id="resume" /> Resume</label>
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
                const msg = JSON.parse(ev.data);
                let text = msg.text || '';
                if (msg.type === 'cycle') text = '═══ Cycle ' + msg.cycle + ' ═══\\n' + text;
                if (msg.type === 'plan') text = 'Plan: ' + text;
                if (msg.type === 'task_done') text = '✓ ' + text;
                if (msg.type === 'task_fail') text = '✗ ' + text;
                if (msg.type === 'test_pass') text = '✓ Tests: ' + text;
                if (msg.type === 'test_fail') text = '✗ Tests: ' + text;
                if (msg.type === 'summary') text = 'Summary: ' + text;
                out.textContent += text + '\\n';
                out.scrollTop = out.scrollHeight;
            };
            ws.onclose = () => out.textContent += '\\n--- Connection closed ---\\n';
            ws.send(JSON.stringify({
                goal: document.getElementById('goal').value,
                resume: document.getElementById('resume').checked
            }));
        };
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

        # Create orchestrator with custom client that streams to websocket
        client = LLMClient()
        orchestrator = Orchestrator(goal=goal, client=client, resume=resume)

        # Run orchestrator with streaming updates
        await orchestrator.run()
        await websocket.send_json({"type": "summary", "text": "Goal complete!"})

    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        await websocket.send_json({"type": "error", "text": str(e)})
    finally:
        manager.disconnect(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
