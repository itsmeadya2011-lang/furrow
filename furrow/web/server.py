from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console
from rich.text import Text

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


class WebSocketConsole:
    """A console-like object that sends output via WebSocket."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self._console = Console(record=True, width=120)

    def print(self, *args, **kwargs) -> None:
        # Use Rich's console to render, then send via websocket
        self._console.print(*args, **kwargs)
        # Get the rendered text
        text = self._console.export_text(styles=True)
        # Send only the new content (last line essentially)
        if text.strip():
            asyncio.create_task(self._send(text))

    async def _send(self, text: str) -> None:
        try:
            await self.websocket.send_text(text)
        except Exception:
            pass


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
        #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; white-space: pre-wrap; max-height: 60vh; overflow: auto; }
        .cycle { color: #64b5f6; font-weight: bold; }
        .success { color: #81c784; }
        .error { color: #e57373; }
        .warning { color: #ffb74d; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <form id="form">
        <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
        <button type="submit">Start</button>
    </form>
    <pre id="out"></pre>
    <script>
        const form = document.getElementById('form');
        const out = document.getElementById('out');
        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value;
            out.textContent = '';
            const ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                const line = ev.data;
                const div = document.createElement('div');
                if (line.includes('═══ Cycle')) div.className = 'cycle';
                else if (line.includes('completed') || line.includes('passed')) div.className = 'success';
                else if (line.includes('failed') || line.includes('Failed') || line.includes('Error')) div.className = 'error';
                else if (line.includes('Warning') || line.includes('warning')) div.className = 'warning';
                div.textContent = line;
                out.appendChild(div);
                out.scrollTop = out.scrollHeight;
            };
            ws.onclose = () => {
                const div = document.createElement('div');
                div.textContent = '\n--- Connection closed ---';
                out.appendChild(div);
            };
            ws.onerror = (err) => {
                const div = document.createElement('div');
                div.className = 'error';
                div.textContent = 'WebSocket error: ' + err;
                out.appendChild(div);
            };
            ws.send(JSON.stringify({goal: goal}));
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
        goal = data.get("goal", "")

        # Create custom console that forwards to websocket
        ws_console = WebSocketConsole(websocket)

        # Create orchestrator with custom console
        orchestrator = Orchestrator(goal=goal, console=ws_console)

        await orchestrator.run()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
