from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class _ClientDisconnected(Exception):
    """Internal signal: the websocket client went away; abort the run."""


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return
    except ValueError:
        data = {}

    payload = data if isinstance(data, dict) else {}
    goal = str(payload.get("goal", "")).strip()
    if not goal:
        await _safe_send(websocket, {"event": "error", "error": "Goal is required."})
        return

    model = payload.get("model")
    if model:
        # A bare model override applies to every agent.
        settings.model = str(model)
        settings.planner_model = str(model)
        settings.worker_model = str(model)
        settings.tester_model = str(model)

    orchestrator = Orchestrator(goal=goal)

    async def send(event: dict[str, Any]) -> None:
        await _safe_send(websocket, event)

    orchestrator.on(send)
    try:
        await orchestrator.run()
    except _ClientDisconnected:
        return  # client closed the socket mid-run
    except Exception as exc:  # noqa: BLE001 - report any orchestrator error to the UI
        await _safe_send(websocket, {"event": "error", "error": str(exc)})
    finally:
        await _safe_send(websocket, {"event": "summary", "summary": orchestrator.summary()})


async def _safe_send(websocket: WebSocket, data: dict[str, Any]) -> None:
    """Send a JSON payload, translating a dead socket into _ClientDisconnected."""
    try:
        await websocket.send_json(data)
    except (WebSocketDisconnect, RuntimeError) as exc:
        raise _ClientDisconnected from exc


def run(host: str | None = None, port: int | None = None) -> None:
    """Run the Furrow web UI, defaulting to Settings (FURROW_WEB_HOST/PORT)."""
    uvicorn.run(app, host=host or settings.web_host, port=port if port is not None else settings.web_port)


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Furrow</title>
    <style>
        :root {
            color-scheme: dark;
        }
        * {
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
                sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        header {
            margin-bottom: 20px;
        }
        h1 {
            margin: 0;
            font-size: 28px;
            color: #58a6ff;
        }
        .subtitle {
            margin: 4px 0 0 0;
            font-size: 13px;
            color: #8b949e;
        }
        #form {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 16px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
        }
        #form label {
            font-size: 13px;
            color: #8b949e;
        }
        #goal {
            flex: 1 1 300px;
            min-width: 200px;
            background-color: #0d1117;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 8px 12px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }
        #model {
            flex: 0 1 200px;
            background-color: #0d1117;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 13px;
        }
        #start {
            background-color: #238636;
            color: #ffffff;
            border: 1px solid #2ea043;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
        }
        #start:disabled {
            background-color: #161b22;
            color: #8b949e;
            border: 1px solid #30363d;
            cursor: not-allowed;
        }
        #status {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 16px;
            font-size: 13px;
            color: #8b949e;
        }
        #log {
            background-color: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px;
            height: calc(100vh - 240px);
            min-height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 12px;
            white-space: pre-wrap;
            word-break: break-word;
        }
        #log .cycle {
            color: #58a6ff;
            font-weight: bold;
            margin: 12px 0 8px 0;
        }
        #log .plan {
            margin: 8px 0;
        }
        #log .task-start {
            color: #d2a8ff;
            margin: 4px 0;
        }
        #log .task-complete {
            color: #3fb950;
            margin: 4px 0;
        }
        #log .task-failed {
            color: #f85149;
            margin: 4px 0;
        }
        #log .test-pass {
            color: #3fb950;
            margin: 4px 0;
        }
        #log .test-fail {
            color: #f85149;
            margin: 4px 0;
        }
        #log .failure-line {
            color: #ffa198;
            margin: 2px 0 2px 16px;
        }
        #log .complete {
            color: #3fb950;
            font-weight: bold;
            margin: 8px 0;
        }
        #log .max-cycles {
            color: #d29922;
            margin: 8px 0;
        }
        #log .error {
            color: #f85149;
            margin: 4px 0;
        }
        #log .summary {
            margin: 8px 0;
        }
        #log .unknown {
            color: #8b949e;
            margin: 4px 0;
        }
    </style>
</head>
<body>
    <header>
        <h1>Furrow</h1>
        <p class="subtitle">AI Agent Orchestrator - Live Progress</p>
    </header>

    <form id="form">
        <label for="goal">Goal</label>
        <input type="text" id="goal" name="goal" placeholder="Enter your goal..." required>
        <label for="model">Model (optional)</label>
        <input type="text" id="model" name="model" placeholder="e.g., gpt-4o">
        <button type="submit" id="start">Start</button>
    </form>

    <div id="status">Idle</div>
    <div id="log"></div>

    <script>
        const form = document.getElementById('form');
        const goalInput = document.getElementById('goal');
        const modelInput = document.getElementById('model');
        const startBtn = document.getElementById('start');
        const statusEl = document.getElementById('status');
        const logEl = document.getElementById('log');

        let ws = null;
        let currentStatus = 'Idle';

        function esc(s) {
            const str = String(s);
            return str
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        }

        function setStatus(text) {
            currentStatus = text;
            statusEl.textContent = text;
        }

        function appendLog(html) {
            logEl.insertAdjacentHTML('beforeend', html + '\n');
            logEl.scrollTop = logEl.scrollHeight;
        }

        function clearLog() {
            logEl.innerHTML = '';
        }

        function disableStart() {
            startBtn.disabled = true;
        }

        function enableStart() {
            startBtn.disabled = false;
        }

        function connect() {
            const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
            ws = new WebSocket(wsUrl);

            ws.onmessage = function(ev) {
                try {
                    const e = JSON.parse(ev.data);
                    render(e);
                } catch (err) {
                    appendLog('<div class="unknown">' + esc('Failed to parse message: ' + err) + '</div>');
                }
            };

            ws.onclose = function() {
                if (currentStatus === 'Running…') {
                    setStatus('Stopped');
                }
                enableStart();
            };

            ws.onerror = function() {
                setStatus('Error');
                enableStart();
            };
        }

        function render(e) {
            const event = e.event;

            switch (event) {
                case 'cycle_start':
                    appendLog('<div class="cycle">═══ Cycle ' + esc(e.cycle ?? '') + ' ═══</div>');
                    break;
                case 'plan':
                    appendLog('<div class="plan"><pre>' + esc(JSON.stringify(e.plan, null, 2)) + '</pre></div>');
                    break;
                case 'task_start':
                    appendLog('<div class="task-start">▶ Task ' + esc(e.task?.id ?? '') + ': ' + esc(e.task?.description ?? '') + '</div>');
                    break;
                case 'task_complete':
                    appendLog('<div class="task-complete">✔ Task ' + esc(e.task?.id ?? '') + ' completed</div>');
                    break;
                case 'task_failed':
                    appendLog('<div class="task-failed">✖ Task ' + esc(e.task?.id ?? '') + ' failed: ' + esc(e.error ?? e.task?.result ?? '') + '</div>');
                    break;
                case 'test_result':
                    if (e.result) {
                        const summary = e.result.summary ?? '';
                        const passed = e.result.passed ?? false;
                        if (passed) {
                            appendLog('<div class="test-pass">Tests passed: ' + esc(summary) + '</div>');
                        } else {
                            appendLog('<div class="test-fail">Tests failed: ' + esc(summary) + '</div>');
                            if (e.result.failures && Array.isArray(e.result.failures)) {
                                for (let i = 0; i < e.result.failures.length; i++) {
                                    appendLog('<div class="failure-line">' + esc(e.result.failures[i]) + '</div>');
                                }
                            }
                        }
                    } else {
                        appendLog('<div class="unknown">' + esc(JSON.stringify(e)) + '</div>');
                    }
                    break;
                case 'complete':
                    appendLog('<div class="complete">Goal complete.</div>');
                    setStatus('Complete');
                    enableStart();
                    break;
                case 'max_cycles_reached':
                    appendLog('<div class="max-cycles">Reached max cycles (' + esc(e.max_cycles ?? '') + ').</div>');
                    setStatus('Complete');
                    enableStart();
                    break;
                case 'error':
                    appendLog('<div class="error">Error: ' + esc(e.error ?? '') + '</div>');
                    setStatus('Error');
                    enableStart();
                    break;
                case 'summary':
                    appendLog('<div class="summary"><pre>' + esc(JSON.stringify(e.summary, null, 2)) + '</pre></div>');
                    break;
                default:
                    appendLog('<div class="unknown">' + esc(JSON.stringify(e)) + '</div>');
                    break;
            }
        }

        form.addEventListener('submit', function(e) {
            e.preventDefault();

            const goal = goalInput.value.trim();
            if (!goal) {
                return;
            }

            const model = modelInput.value.trim();
            const payload = { goal: goal };
            if (model) {
                payload.model = model;
            }

            clearLog();
            setStatus('Running…');
            disableStart();

            if (!ws || ws.readyState !== WebSocket.OPEN) {
                connect();
                ws.addEventListener('open', function onOpen() {
                    ws.send(JSON.stringify(payload));
                    ws.removeEventListener('open', onOpen);
                }, { once: true });
            } else {
                ws.send(JSON.stringify(payload));
            }
        });

        connect();
    </script>
</body>
</html>
"""
