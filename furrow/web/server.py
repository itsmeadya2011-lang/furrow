from __future__ import annotations

import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None


# Store active websocket connections for streaming
_active_websocket: Optional[WebSocket] = None


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #out { background: #1a1a1a; color: #0f0; padding: 15px; border-radius: 8px; height: 400px; overflow-y: auto; white-space: pre-wrap; font-family: monospace; }
        input { width: 70%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; background: #0066cc; color: white; border: none; cursor: pointer; }
        button:hover { background: #0052a3; }
        .status { margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Furrow - Autonomous Coding Agent</h1>
    <form id="form">
        <input id="goal" placeholder="Enter your coding goal..." required />
        <button type="submit">Start</button>
    </form>
    <div id="status" class="status"></div>
    <pre id="out"></pre>
    <script>
        const form = document.getElementById('form');
        const out = document.getElementById('out');
        const status = document.getElementById('status');
        let ws = null;

        form.onsubmit = async (e) => {
            e.preventDefault();
            const goal = document.getElementById('goal').value;
            out.textContent = '';
            status.textContent = 'Connecting...';

            if (ws) ws.close();
            ws = new WebSocket('ws://' + location.host + '/ws');

            ws.onopen = () => {
                status.textContent = 'Running...';
                ws.send(JSON.stringify({goal: goal}));
            };

            ws.onmessage = (ev) => {
                out.textContent += ev.data + '\\n';
                out.scrollTop = out.scrollHeight;
            };

            ws.onclose = () => {
                status.textContent = 'Completed';
            };

            ws.onerror = () => {
                status.textContent = 'Error occurred';
            };
        };
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    global _active_websocket
    await websocket.accept()
    _active_websocket = websocket

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")
        if not goal:
            await websocket.send_text("No goal provided.")
            return

        async def send_output(message: str) -> None:
            """Callback to send output to the websocket."""
            try:
                await websocket.send_text(message)
            except Exception:
                pass  # WebSocket may have closed

        orchestrator = Orchestrator(goal=goal, on_output=send_output)
        await orchestrator.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception:
            pass
    finally:
        _active_websocket = None


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
