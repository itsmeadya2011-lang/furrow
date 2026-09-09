from __future__ import annotations

import asyncio
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator


app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: Optional[str] = None
    max_cycles: Optional[int] = None
    max_parallel: Optional[int] = None
    provider: Optional[str] = None


class WebSocketConsole(Console):
    """Console subclass that captures output via rich's record buffer and
    streams it through the WebSocket as plain text."""

    def __init__(self, websocket: WebSocket, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, record=True, width=120, force_terminal=False)
        self._websocket = websocket

    def print(self, *args: Any, **kwargs: Any) -> None:
        super().print(*args, **kwargs)
        text = self.export_text(clear=True)
        if text:
            try:
                WebSocketConsole._send_text(self._websocket, text)
            except RuntimeError:
                pass

    @staticmethod
    def _send_text(websocket: WebSocket, text: str) -> None:
        """Schedule a WebSocket send. Caller must be in an async context."""
        asyncio.get_running_loop().create_task(websocket.send_text(text))


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 1000px; margin: 0 auto; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
        h1 { color: #569cd6; }
        .status { padding: 10px; margin-bottom: 20px; border-radius: 4px; background: #2d2d2d; border: 1px solid #3c3c3c; }
        .status.running { border-color: #4ec9b0; background: #1b2d2b; }
        .status.complete { border-color: #6a9955; background: #1b2d1b; }
        .status.error { border-color: #f44747; background: #2d1b1b; }
        .status-label { font-weight: bold; color: #9cdcfe; }
        .status-value { color: #ce9178; }
        form { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        input[type="text"] { flex: 1; min-width: 300px; padding: 8px; background: #3c3c3c; border: 1px solid #555; color: #d4d4d4; border-radius: 4px; }
        select { padding: 8px; background: #3c3c3c; border: 1px solid #555; color: #d4d4d4; border-radius: 4px; }
        button { padding: 8px 16px; background: #0e639c; border: none; color: white; border-radius: 4px; cursor: pointer; }
        button:hover { background: #1177bb; }
        button:disabled { background: #555; cursor: not-allowed; }
        #out { background: #1e1e1e; border: 1px solid #3c3c3c; padding: 15px; height: 500px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; }
        .log-line { margin: 2px 0; }
        .log-info { color: #9cdcfe; }
        .log-success { color: #4ec9b0; }
        .log-warning { color: #dcdcaa; }
        .log-error { color: #f44747; }
        .log-dim { color: #6a9955; }
    </style>
</head>
<body>
    <h1>Furrow</h1>
    <div class="status" id="status">
        <span class="status-label">Status:</span> <span class="status-value" id="status-text">Ready</span>
        <span class="status-label" style="margin-left: 20px;">Cycle:</span> <span class="status-value" id="cycle-count">0</span>
    </div>
    <form id="form">
        <input type="text" id="goal" placeholder="Enter goal" required />
        <select id="provider">
            <option value="">Provider (default)</option>
            <option value="anthropic">Anthropic</option>
            <option value="openai">OpenAI</option>
            <option value="ollama">Ollama</option>
        </select>
        <input type="number" id="max-cycles" placeholder="Max cycles" min="0" value="0" style="width: 100px;" />
        <input type="number" id="max-parallel" placeholder="Max parallel" min="1" value="5" style="width: 100px;" />
        <button type="submit" id="start-btn">Start</button>
        <button type="button" id="stop-btn" disabled>Stop</button>
    </form>
    <pre id="out"></pre>
    <script>
        const form = document.getElementById('form');
        const out = document.getElementById('out');
        const statusDiv = document.getElementById('status');
        const statusText = document.getElementById('status-text');
        const cycleCount = document.getElementById('cycle-count');
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        let ws = null;

        function setStatus(state, text) {
            statusDiv.className = 'status ' + state;
            statusText.textContent = text;
        }

        function appendLog(line) {
            const div = document.createElement('div');
            div.className = 'log-line';
            if (line.startsWith('[green]') || line.includes('passed') || line.includes('completed')) {
                div.classList.add('log-success');
            } else if (line.startsWith('[yellow]') || line.includes('warning') || line.includes('batching')) {
                div.classList.add('log-warning');
            } else if (line.startsWith('[red]') || line.includes('failed') || line.includes('Error')) {
                div.classList.add('log-error');
            } else if (line.startsWith('[dim]') || line.startsWith('[')) {
                div.classList.add('log-dim');
            } else {
                div.classList.add('log-info');
            }
            div.textContent = line.replace(/\[\\/?\\w+\]/g, '');
            out.appendChild(div);
            out.scrollTop = out.scrollHeight;
        }

        form.onsubmit = async (e) => {
            e.preventDefault();
            startBtn.disabled = true;
            stopBtn.disabled = false;
            out.innerHTML = '';
            setStatus('running', 'Running...');
            cycleCount.textContent = '0';

            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                const lines = ev.data.split('\\n').filter(l => l.trim());
                lines.forEach(appendLog);
                if (ev.data.includes('cycle.start')) {
                    const match = ev.data.match(/cycle=(\\d+)/);
                    if (match) cycleCount.textContent = match[1];
                }
                if (ev.data.includes('goal.complete')) {
                    setStatus('complete', 'Complete');
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                }
                if (ev.data.includes('max_cycles.reached')) {
                    setStatus('complete', 'Max cycles reached');
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                }
            };
            ws.onclose = () => {
                if (statusText.textContent === 'Running...') {
                    setStatus('error', 'Disconnected');
                }
                startBtn.disabled = false;
                stopBtn.disabled = true;
            };
            ws.onerror = () => {
                setStatus('error', 'Connection error');
                startBtn.disabled = false;
                stopBtn.disabled = true;
            };

            ws.send(JSON.stringify({
                goal: document.getElementById('goal').value,
                provider: document.getElementById('provider').value || null,
                max_cycles: parseInt(document.getElementById('max-cycles').value) || 0,
                max_parallel: parseInt(document.getElementById('max-parallel').value) || 5
            }));
        };

        stopBtn.onclick = () => {
            if (ws) {
                ws.close();
            }
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
        provider = data.get("provider")
        max_cycles = data.get("max_cycles")
        max_parallel = data.get("max_parallel")

        settings = Settings()
        if provider:
            settings.provider = Provider(provider)
        if max_cycles is not None:
            settings.max_cycles = max_cycles
        if max_parallel is not None:
            settings.max_parallel_tasks = max_parallel

        ws_console = WebSocketConsole(websocket)
        orchestrator = Orchestrator(goal=goal, settings=settings, console=ws_console)

        await websocket.send_text(f"[green]Starting orchestrator for goal: {goal}[/green]\n")

        task = asyncio.create_task(orchestrator.run())
        try:
            await task
            await websocket.send_text("[green]Orchestrator completed.[/green]\n")
        except asyncio.CancelledError:
            await websocket.send_text("[yellow]Orchestrator cancelled.[/yellow]\n")
        except WebSocketDisconnect:
            task.cancel()
            raise
        except Exception as e:
            await websocket.send_text(f"[red]Orchestrator error: {e}[/red]\n")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"[red]Error: {e}[/red]\n")
        except Exception:
            pass


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)