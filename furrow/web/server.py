from __future__ import annotations

import asyncio
import logging

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

app = FastAPI(title="Furrow")

# Global orchestrator instance for the web UI
_orchestrator: Orchestrator | None = None
_orchestrator_lock = asyncio.Lock()


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


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
        button:disabled { background: #ccc; cursor: not-allowed; }
        #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; max-height: 60vh; overflow: auto; white-space: pre-wrap; }
        .cycle { border-top: 1px solid #333; margin-top: 1rem; padding-top: 1rem; }
        .task { margin: 0.5rem 0; padding: 0.5rem; background: #2d2d2d; border-radius: 4px; }
        .task.completed { border-left: 4px solid #4CAF50; }
        .task.failed { border-left: 4px solid #f44336; }
        .task.pending { border-left: 4px solid #ff9800; }
        .status-bar { display: flex; gap: 1rem; margin-bottom: 1rem; padding: 0.5rem; background: #2d2d2d; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <div class="status-bar">
        <span id="status">Idle</span>
        <span id="cycle">Cycle: 0</span>
        <span id="progress">Tasks: 0/0</span>
    </div>
    <form id="form">
        <input id="goal" placeholder="Enter goal" required />
        <button type="submit" id="btn">Start</button>
    </form>
    <pre id="out"></pre>
    <script>
        const form = document.getElementById('form');
        const out = document.getElementById('out');
        const btn = document.getElementById('btn');
        const statusEl = document.getElementById('status');
        const cycleEl = document.getElementById('cycle');
        const progressEl = document.getElementById('progress');
        
        let ws = null;
        
        function connect(goal) {
            out.textContent = '';
            statusEl.textContent = 'Connecting...';
            btn.disabled = true;
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onopen = () => {
                statusEl.textContent = 'Running';
                ws.send(JSON.stringify({goal: goal}));
            };
            ws.onmessage = (ev) => {
                try {
                    const data = JSON.parse(ev.data);
                    if (data.type === 'status') {
                        updateStatus(data);
                    } else if (data.type === 'log') {
                        out.textContent += data.message + '\\n';
                        out.scrollTop = out.scrollHeight;
                    } else if (data.type === 'done') {
                        statusEl.textContent = 'Complete';
                        btn.disabled = false;
                    } else if (data.type === 'error') {
                        statusEl.textContent = 'Error';
                        out.textContent += '\\nERROR: ' + data.message + '\\n';
                        btn.disabled = false;
                    }
                } catch {
                    out.textContent += ev.data + '\\n';
                    out.scrollTop = out.scrollHeight;
                }
            };
            ws.onclose = () => {
                statusEl.textContent = 'Disconnected';
                btn.disabled = false;
            };
            ws.onerror = () => {
                statusEl.textContent = 'Connection Error';
                btn.disabled = false;
            };
        }
        
        function updateStatus(data) {
            cycleEl.textContent = 'Cycle: ' + data.cycles;
            progressEl.textContent = 'Tasks: ' + data.completed_count + '/' + data.total_tasks;
            
            // Update task list
            const taskHtml = data.tasks.map(t => 
                '<div class="task ' + t.status + '">' +
                '<strong>Task ' + t.id + ':</strong> ' + t.description +
                (t.result ? '<br><small>' + t.result.substring(0, 200) + '</small>' : '') +
                '</div>'
            ).join('');
            out.innerHTML = '<div class="cycle"><strong>Cycle ' + data.cycles + '</strong></div>' + taskHtml;
            out.scrollTop = out.scrollHeight;
        }
        
        form.onsubmit = (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value;
            if (ws) ws.close();
            connect(goal);
        };
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    global _orchestrator
    await websocket.accept()
    
    async def send_status():
        if _orchestrator:
            status = _orchestrator.get_status()
            await websocket.send_json({"type": "status", **status})
    
    async def send_log(message: str):
        await websocket.send_json({"type": "log", "message": message})
    
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        model = data.get("model")
        
        async with _orchestrator_lock:
            if _orchestrator is None:
                from furrow.llm import LLMClient
                client = LLMClient()
                if model:
                    from furrow.config import settings
                    settings.model = model
                _orchestrator = Orchestrator(goal=goal, client=client)
            else:
                # Update goal for existing orchestrator
                _orchestrator.goal = goal
                _orchestrator.original_goal = goal
        
        # Run orchestrator with status updates
        import furrow.core.orchestrator as orch_module
        original_print = orch_module.console.print
        
        def websocket_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            asyncio.create_task(send_log(msg))
            original_print(*args, **kwargs)
        
        import furrow.core.orchestrator as orch_module
        orch_module.console.print = websocket_print
        
        try:
            await _orchestrator.run()
            await websocket.send_json({"type": "done", "message": "Goal complete"})
        except asyncio.CancelledError:
            await websocket.send_json({"type": "error", "message": "Cancelled"})
            raise
        except Exception as e:
            logger.exception("Orchestrator error")
            await websocket.send_json({"type": "error", "message": str(e)})
        finally:
            orch_module.console.print = original_print
            await send_status()
            
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("WebSocket error")
        await websocket.send_json({"type": "error", "message": str(e)})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)