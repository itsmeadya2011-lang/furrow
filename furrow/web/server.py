from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator, OrchestratorState

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket) -> None:
        await websocket.send_text(message)

    async def broadcast(self, message: str) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass


manager = ConnectionManager()


# Custom console that sends output to WebSocket
class WebSocketConsole:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket

    def print(self, *args, **kwargs) -> None:
        message = " ".join(str(arg) for arg in args)
        asyncio.create_task(self.websocket.send_text(message))

    def log(self, *args, **kwargs) -> None:
        self.print(*args, **kwargs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    for ws in manager.active_connections:
        await ws.close()


app = FastAPI(title="Furrow", lifespan=lifespan)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; 
               white-space: pre-wrap; font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
               max-height: 70vh; overflow-y: auto; }
        .green { color: #4ec9b0; }
        .red { color: #f44747; }
        .yellow { color: #dcdcaa; }
        .cyan { color: #9cdcfe; }
        .blue { color: #569cd6; }
        form { margin-bottom: 1rem; }
        input { padding: 0.5rem; width: 70%; }
        button { padding: 0.5rem 1rem; }
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
        form.onsubmit = async (e) => {
            e.preventDefault();
            out.textContent = '';
            const ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
            ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
            ws.onerror = (err) => out.textContent += '\\n[Error: ' + err + ']\\n';
            ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
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
        if not goal:
            await websocket.send_text("[Error] No goal provided")
            return

        await websocket.send_text(f"[bold green]Furrow[/bold green]\nGoal: {goal}")

        # Create orchestrator with custom output to websocket
        orchestrator = Orchestrator(goal=goal)
        
        # Run orchestrator and capture output
        # We need to monkey-patch the console for this websocket
        import furrow.core.orchestrator as orch_module
        original_console = orch_module.console
        
        class WSConsole:
            def __init__(self, ws):
                self.ws = ws
            def print(self, *args, **kwargs):
                msg = " ".join(str(a) for a in args)
                asyncio.create_task(self.ws.send_text(msg))
            def log(self, *args, **kwargs):
                self.print(*args, **kwargs)
        
        ws_console = WSConsole(websocket)
        orch_module.console = ws_console
        
        try:
            await orchestrator.run()
            
            # Send final summary
            state = orchestrator.get_state()
            summary = {
                "type": "summary",
                "cycles": state.cycle,
                "total_tasks": len(state.all_tasks),
                "completed": len(state.completed_tasks),
                "failed": len(state.failed_tasks),
                "cost": orchestrator.client.get_cost_summary() if hasattr(orchestrator, 'client') else {}
            }
            await websocket.send_text(f"\n[bold cyan]═══ Final Summary ═══[/bold cyan]")
            await websocket.send_text(json.dumps(summary, indent=2))
        finally:
            orch_module.console = original_console
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"[Error] {str(e)}")
    finally:
        manager.disconnect(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)