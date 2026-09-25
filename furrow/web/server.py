from __future__ import annotations

import asyncio
import json
import signal
from typing import Optional, Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Active connections tracking
active_connections: Dict[str, WebSocket] = {}
orchestrators: Dict[str, Orchestrator] = {}


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
    body { font-family: monospace; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
    #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; height: 50vh; overflow: auto; white-space: pre-wrap; }
    input { width: 70%; padding: 0.5rem; }
    button { padding: 0.5rem 1rem; }
    .status { margin: 1rem 0; padding: 0.5rem; background: #2d2d2d; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal (e.g., 'Add JWT auth to the API')" required />
    <button type="submit">Start</button>
  </form>
  <div id="status" class="status">Not connected</div>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const status = document.getElementById('status');
    let ws = null;

    function connect(goal) {
      out.textContent = '';
      status.textContent = 'Connecting...';
      ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onopen = () => {
        status.textContent = 'Connected - Starting task...';
        ws.send(JSON.stringify({goal: goal}));
      };
      ws.onmessage = (ev) => {
        out.textContent += ev.data + '\\n';
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => {
        status.textContent = 'Disconnected';
      };
      ws.onerror = (err) => {
        status.textContent = 'Error: ' + err.message;
      };
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value.trim();
      if (!goal) return;
      if (ws) ws.close();
      connect(goal);
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    conn_id = id(websocket)
    active_connections[str(conn_id)] = websocket

    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        if not goal:
            await websocket.send_text("Error: No goal provided")
            await websocket.close()
            return

        # Create orchestrator with unique state file per connection
        state_file = f".furrow_state_{conn_id}.json"
        orchestrator = Orchestrator(goal=goal, settings=settings.model_copy(update={"state_file": state_file}))
        orchestrators[str(conn_id)] = orchestrator

        # Override console output to send to websocket
        original_print = orchestrator.__class__.__module__
        
        # Run orchestrator
        await orchestrator.run()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"Error: {str(e)}")
        except Exception:
            pass
    finally:
        active_connections.pop(str(conn_id), None)
        orchestrators.pop(str(conn_id), None)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    # Handle graceful shutdown
    def signal_handler():
        print("Shutting down...")
        for conn in active_connections.values():
            try:
                asyncio.create_task(conn.close())
            except Exception:
                pass
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass

    uvicorn.run(app, host=host, port=port)
