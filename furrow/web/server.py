from __future__ import annotations

import asyncio
from collections import deque

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except RuntimeError:
                # Connection closed
                pass


manager = ConnectionManager()
event_log: deque = deque(maxlen=100)


def log_event(event_type: str, data: dict):
    event = {"type": event_type, "data": data}
    event_log.append(event)
    asyncio.create_task(manager.broadcast(event))


class MonitoredOrchestrator(Orchestrator):
    async def _cycle(self) -> None:
        log_event("cycle_start", {"cycle": self.cycles + 1})
        await super()._cycle()
        log_event("cycle_end", {"cycle": self.cycles})

    async def _run_worker(self, task, client):
        log_event("worker_start", {"task_id": task.id, "description": task.description})
        result = await WorkerAgent(task=task, client=client, workspace=self.workspace).run()
        log_event("worker_end", {"task_id": task.id, "status": task.status, "result": result[:200]})
        return result


# Monkey-patch the orchestrator to use monitored version
from furrow.agents.worker import WorkerAgent


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow Monitor</title>
    <style>
        body { font-family: monospace; margin: 20px; background: #1e1e1e; color: #d4d4d4; }
        h1 { color: #4ec9b0; }
        #form { margin-bottom: 20px; }
        input { padding: 10px; width: 60%; background: #3c3c3c; border: 1px solid #555; color: #d4d4d4; }
        button { padding: 10px 20px; background: #4ec9b0; border: none; color: #1e1e1e; cursor: pointer; }
        button:hover { background: #3db8a0; }
        #log { 
            background: #252526; border: 1px solid #3c3c3c; padding: 15px; 
            height: 500px; overflow-y: auto; white-space: pre-wrap; 
            font-size: 13px; line-height: 1.5;
        }
        .event { margin: 5px 0; padding: 5px; border-left: 3px solid #555; }
        .cycle_start { border-color: #4ec9b0; }
        .cycle_end { border-color: #ce9178; }
        .worker_start { border-color: #dcdcaa; }
        .worker_end { border-color: #9cdcfe; }
        .test_start { border-color: #c586c0; }
        .test_end { border-color: #4ec9b0; }
        .error { border-color: #f44747; color: #f44747; }
        .timestamp { color: #888; }
        .task_id { color: #4ec9b0; font-weight: bold; }
        .status_completed { color: #4ec9b0; }
        .status_failed { color: #f44747; }
        .status_pending { color: #dcdcaa; }
    </style>
</head>
<body>
    <h1>Furrow Monitor</h1>
    <form id="form">
        <input id="goal" placeholder="Enter goal" required />
        <button type="submit">Start</button>
    </form>
    <div id="log"></div>
    <script>
        const form = document.getElementById('form');
        const log = document.getElementById('log');
        let ws = null;
        
        function addEvent(event) {
            const div = document.createElement('div');
            div.className = 'event ' + event.type;
            const time = new Date().toLocaleTimeString();
            let html = '<span class="timestamp">[' + time + ']</span> ';
            
            if (event.type === 'cycle_start') {
                html += '<span style="color:#4ec9b0">▶ Cycle ' + event.data.cycle + ' started</span>';
            } else if (event.type === 'cycle_end') {
                html += '<span style="color:#ce9178">■ Cycle ' + event.data.cycle + ' completed</span>';
            } else if (event.type === 'worker_start') {
                html += '<span class="task_id">Worker ' + event.data.task_id + '</span> started: ' + event.data.description;
            } else if (event.type === 'worker_end') {
                const statusClass = 'status_' + event.data.status;
                html += '<span class="task_id">Worker ' + event.data.task_id + '</span> ' + 
                    '<span class="' + statusClass + '">' + event.data.status + '</span>: ' + event.data.result;
            } else if (event.type === 'test_start') {
                html += '<span style="color:#c586c0">◆ Testing started</span>';
            } else if (event.type === 'test_end') {
                html += '<span style="color:#4ec9b0">◆ Testing completed</span>: ' + event.data.summary;
            } else {
                html += JSON.stringify(event);
            }
            
            div.innerHTML = html;
            log.appendChild(div);
            log.scrollTop = log.scrollHeight;
        }
        
        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value;
            log.innerHTML = '';
            
            if (ws) ws.close();
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                const event = JSON.parse(ev.data);
                addEvent(event);
            };
            ws.onclose = () => addEvent({type: 'error', data: {message: 'Connection closed'}});
            ws.send(JSON.stringify({goal: goal}));
        };
        
        // Load history on connect
        fetch('/events').then(r => r.json()).then(events => {
            events.forEach(addEvent);
        });
    </script>
</body>
</html>
""")


@app.get("/events")
async def get_events():
    return list(event_log)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        # Send existing events
        for event in event_log:
            await websocket.send_json(event)
        
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        log_event("test_start", {"summary": "Starting test run"})
        orchestrator = MonitoredOrchestrator(goal=goal)
        await orchestrator.run()
        log_event("test_end", {"summary": "Orchestrator completed"})
    except WebSocketDisconnect:
        pass
    except (RuntimeError, ValueError) as e:
        log_event("error", {"message": str(e)})
    finally:
        manager.disconnect(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
