from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, settings
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
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; background: #1a1a1a; color: #e0e0e0; }
        h1 { color: #4CAF50; }
        #form { margin-bottom: 1rem; }
        #goal { width: 70%; padding: 0.5rem; font-size: 1rem; background: #2d2d2d; border: 1px solid #444; color: #e0e0e0; }
        button { padding: 0.5rem 1rem; font-size: 1rem; background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:disabled { background: #555; cursor: not-allowed; }
        #out { background: #121212; color: #d4d4d4; padding: 1rem; border-radius: 4px; white-space: pre-wrap; max-height: 60vh; overflow-y: auto; border: 1px solid #333; }
        .timestamp { color: #888; }
        .event-started { color: #4CAF50; }
        .event-cycle { color: #2196F3; }
        .event-planning { color: #FF9800; }
        .event-task { color: #9C27B0; }
        .event-completed { color: #4CAF50; font-weight: bold; }
        .event-failed { color: #f44336; }
        .event-tests { color: #00BCD4; }
        .event-stopped { color: #FF9800; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <form id="form">
        <input id="goal" placeholder="Enter goal (e.g., Add JWT auth to the API)" required />
        <button type="submit" id="btn">Start</button>
    </form>
    <pre id="out"></pre>
    <script>
        const form = document.getElementById('form');
        const goalInput = document.getElementById('goal');
        const btn = document.getElementById('btn');
        const out = document.getElementById('out');
        let ws = null;
        
        function log(msg, className = '') {
            const time = new Date().toLocaleTimeString();
            const line = document.createElement('div');
            line.className = className;
            line.innerHTML = '<span class="timestamp">[' + time + ']</span> ' + msg;
            out.appendChild(line);
            out.scrollTop = out.scrollHeight;
        }
        
        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = goalInput.value.trim();
            if (!goal) return;
            
            log('> Starting: ' + goal, 'event-started');
            goalInput.disabled = true;
            btn.disabled = true;
            
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                try {
                    const data = JSON.parse(ev.data);
                    handleEvent(data);
                } catch {
                    // Plain text fallback
                    out.textContent += ev.data + '\\n';
                    out.scrollTop = out.scrollHeight;
                }
            };
            ws.onclose = () => {
                log('[Connection closed]', 'event-stopped');
                goalInput.disabled = false;
                btn.disabled = false;
            };
            ws.onerror = (err) => {
                log('[WebSocket error: ' + err + ']', 'event-failed');
            };
            ws.onopen = () => {
                ws.send(JSON.stringify({goal: goal}));
            };
        };
        
        function handleEvent(data) {
            const event = data.event;
            const d = data.data || {};
            
            switch (event) {
                case 'started':
                    log('Furrow started: ' + d.goal, 'event-started');
                    break;
                case 'cycle_start':
                    log('=== Cycle ' + d.cycle + ' ===', 'event-cycle');
                    break;
                case 'planning_start':
                    log('Planning...', 'event-planning');
                    break;
                case 'planning_complete':
                    log('Planned ' + d.tasks + ' tasks', 'event-planning');
                    break;
                case 'no_tasks_planned':
                    log('No tasks planned', 'event-planning');
                    break;
                case 'execution_start':
                    log('Executing ' + d.tasks + ' tasks in parallel...', 'event-task');
                    break;
                case 'task_completed':
                    log('✓ Task ' + d.task_id + ' completed: ' + d.result, 'event-completed');
                    break;
                case 'task_failed':
                    log('✗ Task ' + d.task_id + ' failed: ' + d.error, 'event-failed');
                    break;
                case 'tests_passed':
                    log('✓ Tests passed: ' + d.summary, 'event-tests');
                    break;
                case 'tests_failed':
                    log('✗ Tests failed: ' + d.summary, 'event-failed');
                    if (d.failures) {
                        d.failures.forEach(f => log('  - ' + f, 'event-failed'));
                    }
                    break;
                case 'completed':
                    log('Goal completed in ' + d.cycles + ' cycles!', 'event-completed');
                    break;
                case 'stopped':
                    log('Stopped after ' + d.cycles + ' cycles', 'event-stopped');
                    break;
                case 'max_cycles_reached':
                    log('Max cycles reached', 'event-stopped');
                    break;
                case 'tasks_limited':
                    log('Limited to ' + d.max_parallel + ' of ' + d.planned + ' tasks', 'event-planning');
                    break;
                case 'state_loaded':
                    log('Loaded state: ' + d.tasks + ' tasks, cycle ' + d.cycle, 'event-cycle');
                    break;
                default:
                    log(JSON.stringify(data));
            }
        }
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
        if not goal:
            await websocket.send_json({"event": "error", "data": {"message": "No goal provided"}})
            await websocket.close()
            return
        
        model = data.get("model")
        if model:
            settings.model = model
        
        # Event callback that sends to websocket
        async def on_event(event: str, event_data: dict):
            try:
                await websocket.send_json({"event": event, "data": event_data})
            except Exception:
                pass  # Connection closed
        
        # Run orchestrator with websocket event callback
        orchestrator = Orchestrator(goal=goal, on_event=on_event)
        await orchestrator.run()
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"event": "error", "data": {"message": str(e)}})
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)