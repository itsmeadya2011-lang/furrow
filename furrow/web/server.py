from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


# Store active sessions
active_sessions: dict[str, Orchestrator] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup on shutdown
    for session in active_sessions.values():
        pass


app = FastAPI(title="Furrow", lifespan=lifespan)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Furrow</title>
    <style>
        body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 1rem; background: #1e1e1e; color: #d4d4d4; }
        h1 { color: #4CAF50; }
        #form { margin-bottom: 1rem; }
        #goal { width: 70%; padding: 0.5rem; font-size: 1rem; background: #2d2d2d; color: #d4d4d4; border: 1px solid #444; }
        button { padding: 0.5rem 1rem; font-size: 1rem; background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:disabled { background: #555; cursor: not-allowed; }
        #out { background: #121212; color: #d4d4d4; padding: 1rem; height: 500px; overflow-y: auto; white-space: pre-wrap; border-radius: 4px; border: 1px solid #333; }
        .cycle { color: #4ec9b0; font-weight: bold; }
        .plan { color: #569cd6; }
        .task { color: #ce9178; }
        .success { color: #4ec9b0; }
        .error { color: #f44747; }
        .warning { color: #dcdcaa; }
        .dim { color: #888; }
    </style>
</head>
<body>
    <h1>Furrow Web Interface</h1>
    <form id="form">
        <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
        <button type="submit" id="btn">Start</button>
    </form>
    <pre id="out"></pre>
    <script>
        const form = document.getElementById('form');
        const out = document.getElementById('out');
        const btn = document.getElementById('btn');
        let ws = null;

        function connect() {
            out.textContent += '\\n[Connecting...]\\n';
            ws = new WebSocket('ws://' + location.host + '/ws');
            ws.onmessage = (ev) => {
                try {
                    const data = JSON.parse(ev.data);
                    if (data.type === 'log') {
                        out.textContent += data.message + '\\n';
                        out.scrollTop = out.scrollHeight;
                    }
                } catch {
                    out.textContent += ev.data + '\\n';
                    out.scrollTop = out.scrollHeight;
                }
            };
            ws.onclose = () => {
                out.textContent += '\\n[Disconnected]\\n';
                btn.disabled = false;
            };
            ws.onerror = (err) => {
                out.textContent += '\\n[Connection error]\\n';
            };
        }

        form.onsubmit = async (e) => {
            e.preventDefault();
            btn.disabled = true;
            const goal = document.getElementById('goal').value;
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                connect();
                await new Promise(r => setTimeout(r, 300));
            }
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({goal: goal}));
            } else {
                btn.disabled = false;
            }
        };
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(id(websocket))
    try:
        while True:
            data = await websocket.receive_json()
            goal = data.get("goal", "")
            if not goal:
                await websocket.send_json({"type": "log", "message": "[Error] No goal provided"})
                continue

            await websocket.send_json({"type": "log", "message": f"[Starting Furrow with goal: {goal}]"})

            # Create a custom console that sends output to websocket
            import furrow.core.orchestrator as orch_module

            class WebSocketConsole:
                def __init__(self, ws):
                    self.ws = ws

                def print(self, *args, **kwargs):
                    from io import StringIO

                    from rich.console import Console as RichConsole
                    temp_console = RichConsole(file=StringIO(), force_terminal=True, width=120, markup=True)
                    temp_console.print(*args, **kwargs)
                    content = temp_console.file.getvalue()
                    if content.strip():
                        asyncio.create_task(self.ws.send_json({"type": "log", "message": content.strip()}))

            ws_console = WebSocketConsole(websocket)
            original_console = orch_module.console
            orch_module.console = ws_console

            orchestrator = Orchestrator(goal=goal)
            active_sessions[session_id] = orchestrator

            try:
                await orchestrator.run()
            except Exception as e:
                await websocket.send_json({"type": "log", "message": f"\n[Error: {e}]"})
            finally:
                orch_module.console = original_console
                active_sessions.pop(session_id, None)
                await websocket.send_json({"type": "log", "message": "\n[Session complete]"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "log", "message": f"\n[Error: {e}]"})
        except Exception:
            pass
    finally:
        active_sessions.pop(session_id, None)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)