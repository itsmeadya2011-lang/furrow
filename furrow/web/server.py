from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from furrow.core.orchestrator import Orchestrator

app = FastAPI(title="Furrow")


class StartRequest(BaseModel):
    goal: str
    model: str | None = None


class ConnectionManager:
    def __init__(self):
        self.active_orchestrators: dict[str, Orchestrator] = {}

    def get_orchestrator(self, client_id: str) -> Orchestrator | None:
        return self.active_orchestrators.get(client_id)

    def set_orchestrator(self, client_id: str, orchestrator: Orchestrator) -> None:
        self.active_orchestrators[client_id] = orchestrator

    def remove_orchestrator(self, client_id: str) -> None:
        self.active_orchestrators.pop(client_id, None)


manager = ConnectionManager()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head><title>Furrow</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  #out { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 4px; height: 500px; overflow-y: auto; white-space: pre-wrap; }
  form { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
  input { flex: 1; padding: 0.5rem; }
  button { padding: 0.5rem 1rem; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
  button:disabled { background: #666; cursor: not-allowed; }
  .stats { margin-top: 1rem; padding: 1rem; background: #f5f5f5; border-radius: 4px; }
</style></head>
<body>
  <h1>Furrow</h1>
  <form id="form">
    <input id="goal" placeholder="Enter goal" required />
    <button type="submit" id="startBtn">Start</button>
    <button type="button" id="stopBtn" disabled onclick="stop()">Stop</button>
  </form>
  <pre id="out"></pre>
  <div class="stats" id="stats"></div>
  <script>
    const form = document.getElementById('form');
    const out = document.getElementById('out');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statsDiv = document.getElementById('stats');
    let ws = null;
    let clientId = Math.random().toString(36).substring(7);
    
    function log(msg) {
      out.textContent += msg + '\\n';
      out.scrollTop = out.scrollHeight;
    }
    
    form.onsubmit = async (e) => {
      e.preventDefault();
      const goal = document.getElementById('goal').value;
      if (!goal) return;
      
      log('\\nStarting...\\n');
      startBtn.disabled = true;
      stopBtn.disabled = false;
      
      ws = new WebSocket('ws://' + location.host + '/ws/' + clientId);
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === 'log') {
            log(data.msg);
          } else if (data.type === 'stats') {
            statsDiv.innerHTML = '<strong>Stats:</strong> ' + JSON.stringify(data.stats, null, 2);
          } else if (data.type === 'done') {
            log('\\n' + data.msg);
            startBtn.disabled = false;
            stopBtn.disabled = true;
            ws.close();
          }
        } catch {
          log(ev.data);
        }
      };
      ws.onclose = () => {
        log('\\nConnection closed.\\n');
        startBtn.disabled = false;
        stopBtn.disabled = true;
      };
      ws.onopen = () => {
        ws.send(JSON.stringify({goal, clientId}));
      };
    };
    
    function stop() {
      if (ws) {
        ws.send(JSON.stringify({action: 'stop'}));
        ws.close();
      }
    }
  </script>
</body>
</html>
""")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
    await websocket.accept()
    orchestrator: Orchestrator | None = None
    
    async def send_log(msg: str) -> None:
        await websocket.send_json({"type": "log", "msg": msg})
    
    async def send_stats(stats: dict) -> None:
        await websocket.send_json({"type": "stats", "stats": stats})
    
    async def send_done(msg: str) -> None:
        await websocket.send_json({"type": "done", "msg": msg})
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("action") == "stop":
                if orchestrator:
                    orchestrator.print_stats()
                await send_done("Stopped by user")
                break
            
            goal = data.get("goal", "")
            if not goal:
                await send_log("No goal provided")
                continue
            
            # Create new orchestrator for this session
            orchestrator = Orchestrator(goal=goal)
            manager.set_orchestrator(client_id, orchestrator)
            
            # Monkey-patch console output to send via websocket
            _ = orchestrator.client.stats.__class__.__module__
            
            # Run the orchestrator
            await orchestrator.run()
            
            # Send final stats
            await send_stats(orchestrator.client.get_stats())
            await send_done("Goal complete!")
            
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        await send_log(f"Error: {e}")
    finally:
        manager.remove_orchestrator(client_id)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)