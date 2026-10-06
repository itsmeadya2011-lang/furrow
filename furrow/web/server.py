from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")

# Store active websocket connections
active_connections: list[WebSocket] = []


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


async def broadcast(message: str) -> None:
    """Send message to all connected websocket clients."""
    disconnected = []
    for ws in active_connections:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        active_connections.remove(ws)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  h1 { color: #4CAF50; }
  #form { margin-bottom: 1rem; }
  #goal { width: 70%; padding: 0.5rem; font-size: 1rem; }
  button { padding: 0.5rem 1rem; font-size: 1rem; background: #4CAF50; color: white; border: none; cursor: pointer; }
  button:disabled { background: #ccc; cursor: not-allowed; }
  #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; max-height: 60vh; overflow: auto; white-space: pre-wrap; }
  .cycle { color: #4ec9b0; font-weight: bold; }
  .task-done { color: #4CAF50; }
  .task-fail { color: #f44747; }
  .plan { color: #dcdcaa; }
  .test-pass { color: #4CAF50; }
  .test-fail { color: #f44747; }
</style></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit" id="btn">Start</button>
  </form>
  <pre id="out"></pre>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const btn = document.getElementById('btn');
    form.onsubmit = async (e) => {
      e.preventDefault();
      btn.disabled = true;
      out.textContent += '\\nStarting...\\n';
      const ws = new WebSocket('ws://' + location.host + '/ws');
      ws.onmessage = (ev) => {
        const msg = ev.data;
        if (msg.startsWith('CYCLE:')) {
          out.textContent += '\\n' + msg.slice(6) + '\\n';
        } else if (msg.startsWith('PLAN:')) {
          out.textContent += '  Plan: ' + msg.slice(5) + '\\n';
        } else if (msg.startsWith('TASK_DONE:')) {
          out.textContent += '  ✓ ' + msg.slice(10) + '\\n';
        } else if (msg.startsWith('TASK_FAIL:')) {
          out.textContent += '  ✗ ' + msg.slice(10) + '\\n';
        } else if (msg.startsWith('TEST_PASS:')) {
          out.textContent += '  ✓ Tests: ' + msg.slice(10) + '\\n';
        } else if (msg.startsWith('TEST_FAIL:')) {
          out.textContent += '  ✗ Tests: ' + msg.slice(10) + '\\n';
        } else if (msg.startsWith('DONE:')) {
          out.textContent += '\\n' + msg.slice(5) + '\\n';
          btn.disabled = false;
        } else if (msg.startsWith('ERROR:')) {
          out.textContent += '\\nERROR: ' + msg.slice(6) + '\\n';
          btn.disabled = false;
        } else {
          out.textContent += msg + '\\n';
        }
        out.scrollTop = out.scrollHeight;
      };
      ws.onclose = () => {
        out.textContent += '\\nClosed.\\n';
        btn.disabled = false;
      };
      ws.send(JSON.stringify({goal: document.getElementById('goal').value}));
    };
  </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    active_connections.append(websocket)
    try:
        data = await websocket.receive_json()
        goal = data.get("goal", "")

        # Create orchestrator with websocket callback
        orchestrator = Orchestrator(goal=goal, on_update=broadcast)
        
        async def run_with_updates():
            try:
                await orchestrator.run()
                await broadcast("DONE:Goal complete!")
            except Exception as e:
                await broadcast(f"ERROR:{e!s}")

        await run_with_updates()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
