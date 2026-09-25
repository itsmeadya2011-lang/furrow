from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class WebSocketCallback:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def send(self, message: str) -> None:
        try:
            await self.websocket.send_text(message)
        except Exception:  # noqa: BLE001, S110
            pass


class WebSocketConsole:
    def __init__(self, ws_callback: WebSocketCallback):
        self.ws_callback = ws_callback
    
    def print(self, *args, **kwargs) -> None:  # type: ignore[override]
        msg = " ".join(str(a) for a in args)
        asyncio.create_task(self.ws_callback.send(msg))


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 1rem; }
        h1 { color: #4CAF50; }
        #form { margin-bottom: 1rem; }
        #goal { width: 70%; padding: 0.5rem; font-size: 1rem; }
        button { padding: 0.5rem 1rem; font-size: 1rem; background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; height: 500px; overflow-y: auto; white-space: pre-wrap; border-radius: 4px; }
        .green { color: #4CAF50; }
        .red { color: #f44336; }
        .yellow { color: #ffeb3b; }
        .cyan { color: #00bcd4; }
        .blue { color: #2196f3; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <form id="form">
        <input id="goal" placeholder="Enter goal (e.g., Add JWT auth to the API)" required />
        <button type="submit">Start</button>
    </form>
    <pre id="out"></pre>
    <script>
        const form = document.getElementById('form');
        const out = document.getElementById('out');
        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value;
            out.textContent += '\\n[Starting Furrow with goal: ' + goal + ']\\n';
            const ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => out.textContent += ev.data + '\\n';
            ws.onclose = () => out.textContent += '\\n[Connection closed]\\n';
            ws.onopen = () => ws.send(JSON.stringify({goal: goal}));
        };
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    callback = WebSocketCallback(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        
        # Create orchestrator with custom console that sends to websocket
        
        ws_console = WebSocketConsole(callback)
        
        # Monkey-patch console for this orchestrator run
        import furrow.core.orchestrator as orchestrator_module
        original_console = orchestrator_module.console
        orchestrator_module.console = ws_console  # type: ignore[assignment]
        
        try:
            orchestrator = Orchestrator(goal=goal)
            await orchestrator.run()
        finally:
            orchestrator_module.console = original_console
            
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        await callback.send(f"[Error] {e}")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
