from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging import setup_logging

setup_logging(settings.log_level)


class WebSocketStream:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    def write(self, data: str) -> None:
        asyncio.get_event_loop().create_task(self._send(data))

    async def _send(self, data: str) -> None:
        try:
            await self.websocket.send_text(data)
        except Exception:
            pass


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
    <script src="https://cdn.jsdelivr.net/npm/ansi_up@latest/ansi_up.js"></script>
    <style>
        body { font-family: monospace; background-color: #000; color: #0f0; padding: 20px; }
        #out { white-space: pre-wrap; word-wrap: break-word; }
        a { color: #0ff; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <form id="form">
        <input id="goal" placeholder="Enter goal" required />
        <button type="submit">Start</button>
    </form>
    <div id="out"></div>
    <script>
        const form = document.getElementById('form');
        const out = document.getElementById('out');
        const ansi_up = new AnsiUp();
        form.onsubmit = async (e) => {
            e.preventDefault();
            out.innerHTML += '\\nStarting...<br>';
            const ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                const html = ansi_up.ansi_to_html(ev.data);
                out.innerHTML += html + '<br>';
                out.scrollTop = out.scrollHeight;
            };
            ws.onclose = () => out.innerHTML += '<br>Closed.<br>';
            ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
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
        ws_stream = WebSocketStream(websocket)
        console = Console(file=ws_stream, force_terminal=True)
        orchestrator = Orchestrator(goal=goal, console=console)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)